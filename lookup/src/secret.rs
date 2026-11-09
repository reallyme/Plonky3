// SPDX-FileCopyrightText: 2026 ReallyMe Technologies Limited
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guarded LogUp trace generation for witness-derived prover memory.

use alloc::vec::Vec;
use core::mem::size_of;

use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse_general,
    dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::secret::{ZeroizingOwner, ZeroizingRowMajorMatrix};
use p3_matrix::stack::VerticalPair;
use p3_maybe_rayon::prelude::*;
use zeroize::Zeroize;

use crate::logup::{LogUpGadget, RowEvalContext, combine_tuple};
use crate::protocol::LookupProtocol;
use crate::types::{Lookup, LookupTerminal};

const LOOKUP_INVERSION_CHUNK_ROWS: usize = 1024;
const INVERSION_WORK_FACTOR: usize = 5;
const PREFIX_SCAN_WORK_FACTOR: usize = 2;

/// Stable failures from guarded LogUp permutation generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SecretLookupGenerationError {
    /// A nonempty lookup argument requires at least one main-trace row.
    #[error("guarded lookup main trace is empty")]
    EmptyMainTrace,
    /// The main trace has a zero width or inconsistent backing length.
    #[error("guarded lookup main-trace shape is invalid")]
    InvalidMainShape,
    /// The optional preprocessed trace has invalid geometry.
    #[error("guarded lookup preprocessed-trace shape is invalid")]
    InvalidPreprocessedShape,
    /// A periodic column has no values.
    #[error("guarded lookup periodic column is empty")]
    EmptyPeriodicColumn,
    /// A periodic column length is not a power of two.
    #[error("guarded lookup periodic column period is invalid")]
    InvalidPeriodicColumn,
    /// A periodic column period does not divide the main-trace height.
    #[error("guarded lookup periodic column does not divide the trace")]
    PeriodicColumnHeightMismatch,
    /// The challenge array does not contain exactly one pair per lookup.
    #[error("guarded lookup challenge count is invalid")]
    ChallengeCountMismatch,
    /// Lookup slots are not contiguous and canonical.
    #[error("guarded lookup column order is invalid")]
    LookupColumnMismatch,
    /// A lookup has no tuples or its tuple and multiplicity counts differ.
    #[error("guarded lookup tuple shape is invalid")]
    LookupTupleShapeMismatch,
    /// An exclusive lookup does not carry one flag per tuple.
    #[error("guarded exclusive lookup flag count is invalid")]
    ExclusiveFlagCountMismatch,
    /// An exclusive selector is not boolean.
    #[error("guarded exclusive lookup flag is not boolean")]
    NonBooleanExclusiveFlag,
    /// More than one exclusive selector is active on a row.
    #[error("guarded exclusive lookup has multiple active flags")]
    MultipleActiveExclusiveFlags,
    /// A sampled challenge collides with a witness-derived lookup value.
    #[error("guarded lookup denominator is zero")]
    ZeroDenominator,
    /// A buffer length, offset, or work estimate cannot be represented.
    #[error("guarded lookup dimensions overflow")]
    DimensionOverflow,
    /// A guarded or public protocol buffer could not be reserved.
    #[error("guarded lookup allocation failed")]
    AllocationFailed,
    /// Validated matrix or chunk geometry drifted during generation.
    #[error("guarded lookup internal shape is invalid")]
    InternalShapeMismatch,
}

/// Guarded permutation trace and its protocol-disclosed terminal.
///
/// AIRs without lookups return `(None, None)` and allocate no secret buffer.
pub type SecretLookupTraceResult<EF> = Result<
    (
        Option<ZeroizingRowMajorMatrix<EF>>,
        Option<LookupTerminal<EF>>,
    ),
    SecretLookupGenerationError,
>;

struct LookupGeometry {
    height: usize,
    width: usize,
    denoms_per_row: usize,
    aux_len: usize,
    chunk_span: usize,
    minimum_task_bytes: usize,
}

fn guarded_zero_vec<T>(len: usize) -> Result<ZeroizingOwner<Vec<T>>, SecretLookupGenerationError>
where
    T: PrimeCharacteristicRing + Zeroize,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| SecretLookupGenerationError::AllocationFailed)?;
    values.resize(len, T::ZERO);
    Ok(ZeroizingOwner::new(values))
}

fn guarded_zero_matrix<T>(
    len: usize,
    width: usize,
) -> Result<ZeroizingRowMajorMatrix<T>, SecretLookupGenerationError>
where
    T: Field + Zeroize,
{
    let mut matrix_values = Vec::new();
    matrix_values
        .try_reserve_exact(len)
        .map_err(|_| SecretLookupGenerationError::AllocationFailed)?;
    matrix_values.resize(len, T::ZERO);
    Ok(ZeroizingOwner::new(RowMajorMatrix::new(
        matrix_values,
        width,
    )))
}

fn validate_inputs<F, EF>(
    gadget: &LogUpGadget,
    main: &RowMajorMatrix<F>,
    preprocessed: &Option<RowMajorMatrix<F>>,
    periodic_columns: &[Vec<F>],
    lookups: &[Lookup<F>],
    challenges: &[EF],
) -> Result<LookupGeometry, SecretLookupGenerationError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if main.width == 0 || !main.values.len().is_multiple_of(main.width) {
        return Err(SecretLookupGenerationError::InvalidMainShape);
    }
    let height = main.values.len() / main.width;
    if height == 0 {
        return Err(SecretLookupGenerationError::EmptyMainTrace);
    }

    if let Some(preprocessed) = preprocessed
        && (preprocessed.width == 0
            || !preprocessed.values.len().is_multiple_of(preprocessed.width)
            || preprocessed.values.len() / preprocessed.width != height)
    {
        return Err(SecretLookupGenerationError::InvalidPreprocessedShape);
    }

    for column in periodic_columns {
        if column.is_empty() {
            return Err(SecretLookupGenerationError::EmptyPeriodicColumn);
        }
        if !column.len().is_power_of_two() {
            return Err(SecretLookupGenerationError::InvalidPeriodicColumn);
        }
        if !height.is_multiple_of(column.len()) {
            return Err(SecretLookupGenerationError::PeriodicColumnHeightMismatch);
        }
    }

    let expected_challenges = lookups
        .len()
        .checked_mul(gadget.num_challenges())
        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
    if challenges.len() != expected_challenges {
        return Err(SecretLookupGenerationError::ChallengeCountMismatch);
    }

    let mut denoms_per_row = 0_usize;
    for (expected_column, lookup) in lookups.iter().enumerate() {
        if lookup.column != expected_column {
            return Err(SecretLookupGenerationError::LookupColumnMismatch);
        }
        if lookup.elements.is_empty() || lookup.elements.len() != lookup.multiplicities.len() {
            return Err(SecretLookupGenerationError::LookupTupleShapeMismatch);
        }
        if let Some(flags) = &lookup.flags
            && flags.len() != lookup.elements.len()
        {
            return Err(SecretLookupGenerationError::ExclusiveFlagCountMismatch);
        }
        denoms_per_row = denoms_per_row
            .checked_add(lookup.num_denoms())
            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
    }

    let width = lookups
        .len()
        .checked_add(1)
        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
    let aux_len = height
        .checked_mul(width)
        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
    let chunk_span = LOOKUP_INVERSION_CHUNK_ROWS
        .checked_mul(width)
        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
    let minimum_task_bytes = INVERSION_WORK_FACTOR
        .checked_mul(LOOKUP_INVERSION_CHUNK_ROWS)
        .and_then(|value| value.checked_mul(denoms_per_row))
        .and_then(|value| value.checked_mul(size_of::<EF>()))
        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;

    Ok(LookupGeometry {
        height,
        width,
        denoms_per_row,
        aux_len,
        chunk_span,
        minimum_task_bytes,
    })
}

impl LogUpGadget {
    /// Generate a LogUp permutation trace while retaining every
    /// witness-derived heap allocation under zeroizing ownership.
    pub fn generate_permutation_secret<F, EF>(
        &self,
        main: &RowMajorMatrix<F>,
        preprocessed: &Option<RowMajorMatrix<F>>,
        periodic_columns: &[Vec<F>],
        public_values: &[F],
        lookups: &[Lookup<F>],
        challenges: &[EF],
    ) -> SecretLookupTraceResult<EF>
    where
        F: Field + Zeroize,
        EF: ExtensionField<F> + Zeroize,
    {
        if lookups.is_empty() {
            return Ok((None, None));
        }

        let geometry = validate_inputs(
            self,
            main,
            preprocessed,
            periodic_columns,
            lookups,
            challenges,
        )?;

        let mut lookup_denom_offsets = Vec::new();
        lookup_denom_offsets
            .try_reserve_exact(
                lookups
                    .len()
                    .checked_add(1)
                    .ok_or(SecretLookupGenerationError::DimensionOverflow)?,
            )
            .map_err(|_| SecretLookupGenerationError::AllocationFailed)?;
        lookup_denom_offsets.push(0_usize);
        for lookup in lookups {
            let next = lookup_denom_offsets
                .last()
                .copied()
                .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?
                .checked_add(lookup.num_denoms())
                .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
            lookup_denom_offsets.push(next);
        }

        let mut lookup_challenges = Vec::new();
        lookup_challenges
            .try_reserve_exact(lookups.len())
            .map_err(|_| SecretLookupGenerationError::AllocationFailed)?;
        for lookup in lookups {
            let base = self
                .num_challenges()
                .checked_mul(lookup.column)
                .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
            let alpha = challenges
                .get(base)
                .copied()
                .ok_or(SecretLookupGenerationError::ChallengeCountMismatch)?;
            let beta = challenges
                .get(
                    base.checked_add(1)
                        .ok_or(SecretLookupGenerationError::DimensionOverflow)?,
                )
                .copied()
                .ok_or(SecretLookupGenerationError::ChallengeCountMismatch)?;
            lookup_challenges.push((alpha, beta));
        }

        let max_tuple_width = lookups
            .iter()
            .flat_map(|lookup| lookup.elements.iter())
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        let mut beta_powers = Vec::new();
        beta_powers
            .try_reserve_exact(lookup_challenges.len())
            .map_err(|_| SecretLookupGenerationError::AllocationFailed)?;
        for &(_, beta) in &lookup_challenges {
            let mut powers = Vec::new();
            powers
                .try_reserve_exact(max_tuple_width)
                .map_err(|_| SecretLookupGenerationError::AllocationFailed)?;
            powers.extend(beta.powers().take(max_tuple_width));
            beta_powers.push(powers);
        }

        let mut aux_trace = guarded_zero_matrix::<EF>(geometry.aux_len, geometry.width)?;
        let mut row_totals = guarded_zero_vec::<EF>(geometry.height)?;

        aux_trace
            .values
            .par_chunks_mut(geometry.chunk_span)
            .zip(row_totals.par_chunks_mut(LOOKUP_INVERSION_CHUNK_ROWS))
            .with_min_task_bytes(geometry.minimum_task_bytes)
            .enumerate()
            .try_for_each(
                |(chunk_index, (chunk_aux, chunk_row_totals))| -> Result<
                    (),
                    SecretLookupGenerationError,
                > {
                    let start_row = chunk_index
                        .checked_mul(LOOKUP_INVERSION_CHUNK_ROWS)
                        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                    let num_rows = chunk_aux.len() / geometry.width;
                    let scratch_len = num_rows
                        .checked_mul(geometry.denoms_per_row)
                        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                    let mut local_denoms = guarded_zero_vec::<EF>(scratch_len)?;
                    let mut local_multiplicities = guarded_zero_vec::<F>(scratch_len)?;
                    let mut periodic_row = Vec::new();
                    periodic_row
                        .try_reserve_exact(periodic_columns.len())
                        .map_err(|_| SecretLookupGenerationError::AllocationFailed)?;

                    for local_row in 0..num_rows {
                        let row = start_row
                            .checked_add(local_row)
                            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                        let next_candidate = row
                            .checked_add(1)
                            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                        let next_row = if next_candidate == geometry.height {
                            0
                        } else {
                            next_candidate
                        };

                        periodic_row.clear();
                        for column in periodic_columns {
                            let period_index = row % column.len();
                            periodic_row.push(
                                *column
                                    .get(period_index)
                                    .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?,
                            );
                        }

                        let local_main_row = main
                            .row_slice(row)
                            .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?;
                        let next_main_row = main
                            .row_slice(next_row)
                            .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?;
                        let main_rows = VerticalPair::new(
                            RowMajorMatrixView::new_row(&local_main_row),
                            RowMajorMatrixView::new_row(&next_main_row),
                        );

                        let preprocessed_rows_data = match preprocessed {
                            Some(preprocessed) => Some((
                                preprocessed
                                    .row_slice(row)
                                    .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?,
                                preprocessed
                                    .row_slice(next_row)
                                    .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?,
                            )),
                            None => None,
                        };
                        let preprocessed_rows = match preprocessed_rows_data.as_ref() {
                            Some((local, next)) => VerticalPair::new(
                                RowMajorMatrixView::new_row(local),
                                RowMajorMatrixView::new_row(next),
                            ),
                            None => VerticalPair::new(
                                RowMajorMatrixView::new(&[], 0),
                                RowMajorMatrixView::new(&[], 0),
                            ),
                        };

                        let row_context = RowEvalContext::new(
                            main_rows,
                            preprocessed_rows,
                            public_values,
                            &periodic_row,
                            challenges,
                            geometry.height,
                            row,
                        );
                        let mut offset = local_row
                            .checked_mul(geometry.denoms_per_row)
                            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;

                        for ((lookup, &(alpha, _)), powers) in lookups
                            .iter()
                            .zip(lookup_challenges.iter())
                            .zip(beta_powers.iter())
                        {
                            match &lookup.flags {
                                None => {
                                    for (elements, multiplicity) in
                                        lookup.elements.iter().zip(&lookup.multiplicities)
                                    {
                                        let resolved_multiplicity =
                                            multiplicity.resolve(&row_context);
                                        let denominator = if resolved_multiplicity.is_zero() {
                                            EF::ONE
                                        } else {
                                            alpha
                                                - combine_tuple::<F, EF>(
                                                    elements,
                                                    powers,
                                                    &row_context,
                                                )
                                        };
                                        *local_multiplicities.get_mut(offset).ok_or(
                                            SecretLookupGenerationError::InternalShapeMismatch,
                                        )? = resolved_multiplicity;
                                        *local_denoms.get_mut(offset).ok_or(
                                            SecretLookupGenerationError::InternalShapeMismatch,
                                        )? = denominator;
                                        offset = offset
                                            .checked_add(1)
                                            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                                    }
                                }
                                Some(flags) => {
                                    let mut numerator = F::ZERO;
                                    let mut flag_sum = F::ZERO;
                                    let mut denominator = EF::ZERO;
                                    for ((elements, multiplicity), flag_expression) in lookup
                                        .elements
                                        .iter()
                                        .zip(&lookup.multiplicities)
                                        .zip(flags)
                                    {
                                        let flag = flag_expression.resolve(&row_context);
                                        if !flag.is_zero() && flag != F::ONE {
                                            return Err(
                                                SecretLookupGenerationError::NonBooleanExclusiveFlag,
                                            );
                                        }
                                        flag_sum += flag;
                                        if flag.is_zero() {
                                            continue;
                                        }
                                        numerator += flag * multiplicity.resolve(&row_context);
                                        denominator += (alpha
                                            - combine_tuple::<F, EF>(
                                                elements,
                                                powers,
                                                &row_context,
                                            )) * flag;
                                    }
                                    if !flag_sum.is_zero() && flag_sum != F::ONE {
                                        return Err(
                                            SecretLookupGenerationError::MultipleActiveExclusiveFlags,
                                        );
                                    }
                                    denominator += F::ONE - flag_sum;
                                    *local_multiplicities.get_mut(offset).ok_or(
                                        SecretLookupGenerationError::InternalShapeMismatch,
                                    )? = numerator;
                                    *local_denoms.get_mut(offset).ok_or(
                                        SecretLookupGenerationError::InternalShapeMismatch,
                                    )? = denominator;
                                    offset = offset
                                        .checked_add(1)
                                        .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                                }
                            }
                        }
                        let expected_end = local_row
                            .checked_add(1)
                            .and_then(|value| value.checked_mul(geometry.denoms_per_row))
                            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                        if offset != expected_end {
                            return Err(SecretLookupGenerationError::InternalShapeMismatch);
                        }
                    }

                    if local_denoms.iter().any(Field::is_zero) {
                        return Err(SecretLookupGenerationError::ZeroDenominator);
                    }
                    let mut local_inverses = guarded_zero_vec::<EF>(scratch_len)?;
                    batch_multiplicative_inverse_general(
                        &local_denoms,
                        &mut local_inverses,
                        |value| value.inverse(),
                    );
                    drop(local_denoms);

                    for (local_row, row_total_slot) in
                        chunk_row_totals.iter_mut().enumerate()
                    {
                        let inverse_base = local_row
                            .checked_mul(geometry.denoms_per_row)
                            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                        let row_offset = local_row
                            .checked_mul(geometry.width)
                            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                        let mut row_total = EF::ZERO;
                        for lookup_index in 0..lookups.len() {
                            let start = *lookup_denom_offsets.get(lookup_index).ok_or(
                                SecretLookupGenerationError::InternalShapeMismatch,
                            )?;
                            let end = *lookup_denom_offsets
                                .get(lookup_index.checked_add(1).ok_or(
                                    SecretLookupGenerationError::DimensionOverflow,
                                )?)
                                .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?;
                            let absolute_start = inverse_base
                                .checked_add(start)
                                .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                            let absolute_end = inverse_base
                                .checked_add(end)
                                .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                            let inverses = local_inverses
                                .get(absolute_start..absolute_end)
                                .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?;
                            let multiplicities = local_multiplicities
                                .get(absolute_start..absolute_end)
                                .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?;
                            let fraction = dot_product(
                                inverses.iter().copied(),
                                multiplicities.iter().copied(),
                            );
                            let fraction_column = lookup_index
                                .checked_add(1)
                                .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                            let output_index = row_offset
                                .checked_add(fraction_column)
                                .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
                            *chunk_aux
                                .get_mut(output_index)
                                .ok_or(SecretLookupGenerationError::InternalShapeMismatch)? =
                                fraction;
                            row_total += fraction;
                        }
                        *row_total_slot = row_total;
                        row_total.zeroize();
                    }
                    Ok(())
                },
            )?;

        let num_threads = current_num_threads();
        if num_threads == 0 {
            return Err(SecretLookupGenerationError::InternalShapeMismatch);
        }
        let prefix_task_bytes = PREFIX_SCAN_WORK_FACTOR
            .checked_mul(size_of::<EF>())
            .ok_or(SecretLookupGenerationError::DimensionOverflow)?;
        let chunk_size = geometry
            .height
            .div_ceil(num_threads)
            .max(min_task_len(geometry.height, prefix_task_bytes));
        if chunk_size == 0 {
            return Err(SecretLookupGenerationError::InternalShapeMismatch);
        }

        row_totals.par_chunks_mut(chunk_size).for_each(|chunk| {
            let mut running_total = EF::ZERO;
            for value in chunk {
                running_total += *value;
                *value = running_total;
            }
            running_total.zeroize();
        });

        let offset_count = geometry.height.div_ceil(chunk_size);
        let mut offsets = guarded_zero_vec::<EF>(offset_count)?;
        let mut cumulative_offset = EF::ZERO;
        for (offset, chunk) in offsets.iter_mut().zip(row_totals.chunks(chunk_size)) {
            *offset = cumulative_offset;
            let chunk_total = chunk
                .last()
                .copied()
                .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?;
            cumulative_offset += chunk_total;
        }
        cumulative_offset.zeroize();

        row_totals
            .par_chunks_mut(chunk_size)
            .enumerate()
            .try_for_each(
                |(chunk_index, chunk)| -> Result<(), SecretLookupGenerationError> {
                    let offset = *offsets
                        .get(chunk_index)
                        .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?;
                    if !offset.is_zero() {
                        for value in chunk {
                            *value += offset;
                        }
                    }
                    Ok(())
                },
            )?;

        let terminal = LookupTerminal(
            row_totals
                .last()
                .copied()
                .ok_or(SecretLookupGenerationError::InternalShapeMismatch)?,
        );
        aux_trace
            .values
            .par_chunks_mut(geometry.width)
            .skip(1)
            .zip(row_totals.par_iter())
            .try_for_each(
                |(row, previous_total)| -> Result<(), SecretLookupGenerationError> {
                    *row.get_mut(0)
                        .ok_or(SecretLookupGenerationError::InternalShapeMismatch)? =
                        *previous_total;
                    Ok(())
                },
            )?;

        Ok((Some(aux_trace), Some(terminal)))
    }
}
