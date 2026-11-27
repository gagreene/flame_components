use ndarray::{ArrayD, ArrayViewD, Dimension, IxDyn};

use crate::Error;

/// Evaluate aligned cells in logical row-major order, including strided and 0-D views.
pub(crate) fn broadcast<T: Copy, U, const N: usize>(
    inputs: [ArrayViewD<'_, T>; N],
    mut evaluate: impl FnMut([T; N]) -> U,
) -> Result<ArrayD<U>, Error> {
    broadcast_inner(inputs, None::<ArrayViewD<'_, ()>>, |cells, _| {
        evaluate(cells)
    })
}

/// Broadcast one differently typed input alongside the other inputs.
pub(crate) fn broadcast_with_extra<T: Copy, E: Copy, U, const N: usize>(
    inputs: [ArrayViewD<'_, T>; N],
    extra: ArrayViewD<'_, E>,
    mut evaluate: impl FnMut([T; N], E) -> U,
) -> Result<ArrayD<U>, Error> {
    broadcast_inner(inputs, Some(extra), |cells, extra| {
        evaluate(cells, extra.expect("the extra input was provided"))
    })
}

fn broadcast_inner<T: Copy, E: Copy, U, const N: usize>(
    inputs: [ArrayViewD<'_, T>; N],
    extra: Option<ArrayViewD<'_, E>>,
    mut evaluate: impl FnMut([T; N], Option<E>) -> U,
) -> Result<ArrayD<U>, Error> {
    let shapes = || {
        inputs
            .iter()
            .map(|input| input.shape())
            .chain(extra.iter().map(|input| input.shape()))
    };
    let incompatible = || Error::IncompatibleShapes {
        shapes: shapes().map(<[usize]>::to_vec).collect(),
    };
    let rank = shapes().map(<[usize]>::len).max().unwrap_or(0);
    let mut shape = vec![1; rank];
    for input in shapes() {
        for (output_len, &input_len) in shape.iter_mut().rev().zip(input.iter().rev()) {
            if *output_len == 1 {
                *output_len = input_len;
            } else if input_len != 1 && *output_len != input_len {
                return Err(incompatible());
            }
        }
    }
    let too_large = || Error::OutputTooLarge {
        shape: shape.clone(),
    };
    let dimension = IxDyn(&shape);
    let length = dimension.size_checked().ok_or_else(too_large)?;
    let mut output = Vec::new();
    output.try_reserve_exact(length).map_err(|_| too_large())?;
    let views = inputs
        .iter()
        .map(|input| input.broadcast(dimension.clone()).ok_or_else(incompatible))
        .collect::<Result<Vec<_>, _>>()?;
    let extra_view = extra
        .as_ref()
        .map(|input| input.broadcast(dimension.clone()).ok_or_else(incompatible))
        .transpose()?;
    let mut cells = views.iter().map(|view| view.iter()).collect::<Vec<_>>();
    let mut extra_cells = extra_view.as_ref().map(|view| view.iter());
    for _ in 0..length {
        let values = std::array::from_fn(|index| {
            *cells[index].next().expect("each view has the output shape")
        });
        let extra_value = extra_cells
            .as_mut()
            .map(|cells| *cells.next().expect("the extra view has the output shape"));
        output.push(evaluate(values, extra_value));
    }
    ArrayD::from_shape_vec(dimension, output).map_err(|_| too_large())
}

#[cfg(test)]
mod tests {
    use ndarray::{arr0, arr2, Array0};

    use super::*;

    #[test]
    fn scalar_broadcast_retains_scalar_shape() {
        let input = arr0(3);
        let output = broadcast([input.view().into_dyn()], |[value]| value).unwrap();
        assert_eq!(output, Array0::from_elem((), 3).into_dyn());
    }

    #[test]
    fn empty_axis_broadcasts_with_a_singleton() {
        let empty = arr2::<i32, 0>(&[[], []]).into_dyn();
        let singleton = arr2(&[[7]]).into_dyn();
        let result: ArrayD<i32> =
            broadcast([empty.view(), singleton.view()], |_| unreachable!()).unwrap();
        assert_eq!(result.shape(), &[2, 0]);
    }
}
