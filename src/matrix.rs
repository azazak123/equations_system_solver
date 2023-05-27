#![allow(clippy::needless_range_loop)]
use std::ops::Mul;

use crate::Element;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Matrix<T> {
    pub values: Vec<Vec<T>>,
    pub n: usize,
    pub m: usize,
}

impl<T: Element> Matrix<T> {
    pub fn new(matrix: Vec<Vec<T>>) -> Option<Self> {
        if matrix.is_empty() || matrix[0].is_empty() {
            return None;
        }

        Some(Matrix {
            n: matrix.len(),
            m: matrix[0].len(),
            values: matrix,
        })
    }

    pub fn transpose(&self) -> Self {
        let mut result = vec![vec![T::zero(); self.n]; self.m];

        for i in 0..self.n {
            for j in 0..self.m {
                result[j][i] = self.values[i][j];
            }
        }

        Matrix::new(result).unwrap()
    }

    pub fn det(&self) -> Option<T> {
        if self.n != self.m {
            return None;
        }

        let n = self.n;

        if n == 1 {
            return Some(self.values[0][0]);
        }

        let mut result = T::one();
        let mut matrix = self.values.to_vec();

        for i in 0..n {
            let mut max_row = i;

            for j in (i + 1)..n {
                if matrix[j][i].abs() > matrix[max_row][i].abs() {
                    max_row = j;
                }
            }

            if max_row != i {
                matrix.swap(max_row, i);
                result = result * -T::one();
            }

            let pivot = matrix[i][i];

            if pivot.is_zero() {
                return Some(pivot);
            }

            result = result * pivot;

            for j in (i + 1)..n {
                let factor = matrix[j][i] / pivot;

                for k in i..n {
                    matrix[j][k] = matrix[j][k] - matrix[i][k] * factor;
                }
            }
        }

        Some(result)
    }

    pub fn is_diagonal_row_dominated(&self) -> bool {
        self.values
            .iter()
            .enumerate()
            .all(|(i, arr)| arr[i].abs() > (arr.iter().map(T::abs).sum::<T>() - arr[i].abs()))
    }

    pub fn is_diagonal_col_dominated(&self) -> bool {
        self.clone()
            .transpose()
            .values
            .iter()
            .enumerate()
            .all(|(i, arr)| arr[i].abs() > (arr.iter().map(T::abs).sum::<T>() - arr[i].abs()))
    }

    pub fn norm(&self) -> T {
        self.values
            .iter()
            .map(|arr| arr.iter().cloned().map(|el| el.abs()).sum())
            .reduce(|acc, el| if el > acc { el } else { acc })
            .unwrap()
    }
}

impl<T: Element> Mul<T> for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = vec![vec![T::zero(); self.m]; self.n];

        for i in 0..self.n {
            for j in 0..self.m {
                result[i][j] = rhs * self.values[i][j]
            }
        }

        Matrix::new(result).unwrap()
    }
}

impl<T: Element> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = vec![vec![T::zero(); rhs.m]; self.n];

        for i in 0..self.n {
            for j in 0..rhs.m {
                result[i][j] = self.values[i]
                    .iter()
                    .zip(rhs.transpose().values[j].iter())
                    .map(|(el1, el2)| *el1 * *el2)
                    .sum()
            }
        }

        Matrix::new(result).unwrap()
    }
}

impl<T: Element> Mul<Vec<T>> for Matrix<T> {
    type Output = Vec<T>;

    fn mul(self, rhs: Vec<T>) -> Self::Output {
        let mut result = vec![T::zero(); self.n];

        for i in 0..self.n {
            result[i] = self.values[i]
                .iter()
                .zip(rhs.iter())
                .map(|(el1, el2)| *el1 * *el2)
                .sum()
        }

        result
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn transpose_work() {
        let result = Matrix::new(vec![
            vec![2, 3, 1],
            vec![3, -1, 2],
            vec![40, 5, -4],
            vec![10; 3],
        ])
        .unwrap()
        .transpose();

        let answer = Matrix::new(vec![
            vec![2, 3, 40, 10],
            vec![3, -1, 5, 10],
            vec![1, 2, -4, 10],
        ])
        .unwrap();

        assert_eq!(result, answer);
    }
    #[test]
    fn mul_work() {
        let answer = Matrix::new(vec![vec![2, 3, 1], vec![3, -1, 2], vec![40, 5, -4]])
            .unwrap()
            .transpose();

        let result = answer.clone()
            * Matrix::new(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]).unwrap();

        assert_eq!(result, answer);
    }
}
