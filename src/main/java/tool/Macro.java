package tool;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * common matrix or math operation
 *
 * @author Max Ngai
 * @since 2022/10/11
 */
public class Macro {

    /**
     * vector horizontal stack
     * @param array
     * @return
     */
    public static Array2DRowRealMatrix hstack(double[]... array) {
        int p = array.length;
        int n = array[0].length;

        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(n, p);
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < n; j++) {
                matrix.setEntry(j, i, array[i][j]);
            }
        }

        return matrix;
    }

    public static Array2DRowRealMatrix hstack(List<double[]> arrays) {
        double[][] a = new double[arrays.size()][];

        for (int i = 0; i < arrays.size(); i++) {
            a[i] = arrays.get(i);
        }

        return hstack(a);
    }

    /**
     * vector vertical stack
     * @param array
     * @return
     */
    public static Array2DRowRealMatrix vstack(double[]... array) {
        int n = array.length;
        int p = array[0].length;

        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(n, p);
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < n; j++) {
                matrix.setEntry(j, i, array[j][i]);
            }
        }

        return matrix;
    }

    public static Array2DRowRealMatrix vstack(List<double[]> arrays) {
        double[][] a = new double[arrays.size()][];

        for (int i = 0; i < arrays.size(); i++) {
            a[i] = arrays.get(i);
        }

        return vstack(a);
    }


    public static double[] toArray(List<Double> e) {
        double[] res = new double[e.size()];
        for (int i = 0; i < e.size(); i++) {
            res[i] = e.get(i);
        }
        return res;
    }


    /**
     * 行列式
     * @param x
     * @return
     */
    public static double determinant(double[][] x) {
        if (x.length != x[0].length) {
            throw new IllegalArgumentException("row != column");
        } else if (x.length == 1 && x[0].length == 1) {
            return x[0][0];
        } else if (x.length == 2 && x[0].length == 2) {
            return x[0][0] * x[1][1] - x[1][0] * x[0][1];
        } else {
            Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(x);
            double res = 0D;
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                AtomicInteger removedColumn = new AtomicInteger(j);
                int[] rows = IntStream.range(1, matrix.getRowDimension()).toArray();
                int[] columns = IntStream.range(0, matrix.getColumnDimension()).filter(i -> i != removedColumn.get()).toArray();
                RealMatrix m = matrix.getSubMatrix(rows, columns);
                double e = Math.pow(-1, (1 + j + 1)) * matrix.getEntry(0, j) * determinant(m.getData());
                res += e;
            }
            return res;
        }


    }
}
