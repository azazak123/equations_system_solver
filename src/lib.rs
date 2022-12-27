pub mod element;
mod matrix;

use element::Element;
use matrix::Matrix;
use num::traits::real::Real;

pub fn solve_sqrt<T: Element + Real>(coeffs: &Matrix<T>, answers: &[T]) -> Option<Vec<T>> {
    let determinant = coeffs.det()?;

    if determinant.is_zero() {
        return None;
    }

    let transp_coeffs = coeffs.transpose();
    let coeffs_sym = transp_coeffs.clone() * coeffs.to_owned();
    let answers_sym = transp_coeffs * answers.to_owned();

    let mut s_matrix = vec![vec![T::zero(); coeffs_sym.m]; coeffs_sym.n];

    for i in 0..s_matrix.len() {
        s_matrix[i][i] = (coeffs_sym.values[i][i]
            - s_matrix[..i].iter().map(|arr| arr[i] * arr[i]).sum())
        .sqrt();
        for j in i + 1..s_matrix.len() {
            s_matrix[i][j] = (coeffs_sym.values[i][j]
                - s_matrix[..i].iter().map(|arr| arr[i] * arr[j]).sum())
                / s_matrix[i][i]
        }
    }

    let mut y_vec = vec![T::zero(); answers_sym.len()];

    for i in 0..y_vec.len() {
        y_vec[i] = (answers_sym[i]
            - y_vec[..i]
                .iter()
                .enumerate()
                .map(|(j, el)| s_matrix[j][i] * *el)
                .sum())
            / s_matrix[i][i];
    }

    let mut solution_vec = vec![T::zero(); answers_sym.len()];

    for i in (0..solution_vec.len()).rev() {
        solution_vec[i] = (y_vec[i]
            - solution_vec[i + 1..]
                .iter()
                .zip(i + 1..)
                .map(|(el, j)| s_matrix[i][j] * *el)
                .sum())
            / s_matrix[i][i];
    }

    Some(solution_vec)
}

pub fn solve_iteration<T: Element>(
    coeffs: &Matrix<T>,
    answers: &[T],
    precision: T,
) -> Option<(Vec<T>, u128, T)> {
    let determinant = coeffs.det()?;

    if determinant.is_zero() {
        return None;
    }

    let mut coeffs_dominated = coeffs.clone();
    let mut answers_dominated = answers.to_owned();

    while !coeffs_dominated.is_diagonal_row_dominated() {
        for i in 0..coeffs_dominated.n {
            if coeffs_dominated.values[i][i] < T::zero() {
                coeffs_dominated.values[i] = coeffs_dominated.values[i]
                    .iter()
                    .map(|el| el.neg())
                    .collect();

                answers_dominated[i] = -answers_dominated[i]
            }

            let mut max = coeffs_dominated.values[i][i].abs()
                - (coeffs_dominated.values[i].iter().map(T::abs).sum::<T>()
                    - coeffs_dominated.values[i][i]);

            let max_cl = max;

            let mut max_index = coeffs_dominated.n;
            for j in (0..coeffs_dominated.n)
                .filter(|j| *j != i && coeffs.values[*j][i] > T::zero())
                .filter(|_| max_cl < T::zero())
            {
                let div = (coeffs_dominated.values[i][i] + coeffs.values[j][i]).abs()
                    - coeffs.values[j][0..i]
                        .iter()
                        .chain(coeffs.values[j][i + 1..].iter())
                        .zip(
                            coeffs_dominated.values[i][0..i]
                                .iter()
                                .chain(coeffs_dominated.values[i][i + 1..].iter()),
                        )
                        .map(|(el1, el2)| (*el1 + *el2).abs())
                        .sum();

                if max < div {
                    max = div;
                    max_index = j;
                }
            }

            if max_index != coeffs_dominated.n {
                coeffs_dominated.values[i] = coeffs_dominated.values[i]
                    .iter()
                    .zip(coeffs.values[max_index].iter())
                    .map(|(el1, el2)| *el1 + *el2)
                    .collect();

                answers_dominated[i] = answers_dominated[i] + answers[max_index]
            }
        }
    }

    let mut solution_vec =
        vec![(T::one() + T::one()) * (T::one() + T::one()); answers_dominated.len()];
    let mut prev_solution_vec = vec![T::zero(); solution_vec.len()];

    let mut norm = coeffs_dominated.norm();

    while norm >= T::one() {
        coeffs_dominated.values = coeffs_dominated
            .values
            .iter()
            .map(|arr| {
                arr.iter()
                    .map(|el| *el / ((T::one() + T::one()) * norm))
                    .collect()
            })
            .collect();

        answers_dominated = answers_dominated
            .iter()
            .map(|el| *el / ((T::one() + T::one()) * norm))
            .collect();

        norm = coeffs_dominated.norm();
    }

    let mut bias = norm
        * solution_vec
            .iter()
            .zip(prev_solution_vec.iter())
            .map(|(el, prev_el)| *el - *prev_el)
            .reduce(|acc, el| {
                if el.abs() > acc.abs() {
                    el.abs()
                } else {
                    acc.abs()
                }
            })
            .unwrap()
        / (T::one() - norm);

    let mut k = 0;

    while bias > precision {
        prev_solution_vec = solution_vec.clone();

        for i in 0..solution_vec.len() {
            solution_vec[i] = -prev_solution_vec
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, el)| *el * coeffs_dominated.values[i][j] / coeffs_dominated.values[i][i])
                .sum::<T>()
                + answers_dominated[i] / coeffs_dominated.values[i][i]
        }

        k += 1;

        bias = norm
            * solution_vec
                .iter()
                .zip(prev_solution_vec.iter())
                .map(|(el, prev_el)| *el - *prev_el)
                .reduce(|acc, el| {
                    if el.abs() > acc.abs() {
                        el.abs()
                    } else {
                        acc.abs()
                    }
                })
                .unwrap()
            / (T::one() - norm);
    }

    Some((solution_vec, k, bias))
}

#[cfg(test)]
mod tests {

    use super::*;

    fn init() -> (Matrix<f64>, Vec<f64>, Vec<f64>) {
        (
            Matrix::new(vec![
                vec![2.7, 3.3, 1.3],
                vec![3.5, -1.7, 2.8],
                vec![4.1, 5.8, -1.7],
            ])
            .unwrap(),
            vec![2.1, 1.7, 0.8],
            vec![0.061, 0.304, 0.716],
        )
    }

    #[test]
    fn sqrt_method_work() {
        let (a, f, answer) = init();

        let result = solve_sqrt(&a, f.as_slice()).unwrap();

        assert!(answer
            .iter()
            .zip(result.iter())
            .all(|(a, r)| num::Float::abs(*a - *r) < 0.001));
    }

    #[test]
    fn iteration_method_work() {
        let (a, f, answer) = init();

        let (result, _, _) = solve_iteration(&a, f.as_slice(), 0.001).unwrap();

        assert!(answer
            .iter()
            .zip(result.iter())
            .all(|(a, r)| num::Float::abs(*a - *r) < 0.001));
    }

    #[test]
    fn compare_methods() {
        let (a, f, answer) = init();

        let sqrt = solve_sqrt(&a, f.as_slice()).unwrap();

        let (iteration, k, bias) = solve_iteration(&a, f.as_slice(), 0.001).unwrap();

        let val1: Vec<f64> = (a.clone() * answer.clone())
            .iter()
            .zip(f.iter())
            .map(|(el, f)| el - f)
            .collect();

        let val2: Vec<f64> = (a.clone() * sqrt.clone())
            .iter()
            .zip(f.iter())
            .map(|(el, f)| el - f)
            .collect();

        let val3: Vec<f64> = (a * iteration.clone())
            .iter()
            .zip(f.iter())
            .map(|(el, f)| el - f)
            .collect();

        println!("{:25} {:^28} {:^28} {:^10} {:^15}", "", "x*", "δ", "k", "r");
        println!("{:25} {:^.5?} {:^.5?}", "Online solver", answer, val1);
        println!("{:25} {:^.5?} {:^.5?}", "Sqrt method", sqrt, val2);
        println!(
            "{:25} {:^.5?} {:^.5?} {:^10} {:^15.5}",
            "Iteration method", iteration, val3, k, bias
        );
    }
}
