use ndarray::{ArrayD, ArrayViewD, Dimension, IxDyn};

use crate::Error;

/// Evaluate aligned cells in logical row-major order, including strided and 0-D views.
pub(crate) fn broadcast<T: Copy, U, const N: usize>(
    inputs: [ArrayViewD<'_, T>; N],
    mut evaluate: impl FnMut([T; N]) -> U,
) -> Result<ArrayD<U>, Error> {
    let incompatible = || Error::IncompatibleShapes {
        shapes: inputs.iter().map(|input| input.shape().to_vec()).collect(),
    };
    let rank = inputs.iter().map(|input| input.ndim()).max().unwrap_or(0);
    let mut shape = vec![1; rank];
    for input in &inputs {
        for (output_len, &input_len) in shape.iter_mut().rev().zip(input.shape().iter().rev()) {
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
    let mut cells = views.iter().map(|view| view.iter()).collect::<Vec<_>>();
    for _ in 0..length {
        output.push(evaluate(std::array::from_fn(|index| {
            *cells[index].next().expect("each view has the output shape")
        })));
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
