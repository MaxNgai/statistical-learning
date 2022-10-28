package tool;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

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
    public static Array2DRowRealMatrix matrixHConcat(double[]... array) {
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

    public static Array2DRowRealMatrix matrixHConcat(List<double[]> arrays) {
        double[][] a = new double[arrays.size()][];

        for (int i = 0; i < arrays.size(); i++) {
            a[i] = arrays.get(i);
        }

        return matrixHConcat(a);
    }

    /**
     * vector vertical stack
     * @param array
     * @return
     */
    public static Array2DRowRealMatrix matrixVConcat(double[]... array) {
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

    public static Array2DRowRealMatrix matrixVConcat(List<double[]> arrays) {
        double[][] a = new double[arrays.size()][];

        for (int i = 0; i < arrays.size(); i++) {
            a[i] = arrays.get(i);
        }

        return matrixVConcat(a);
    }


    @Deprecated
    public static ArrayRealVector tStatistics(OLSMultipleLinearRegression rg) {

        return RegressionUtil.tStatistics(rg);
    }

    public static double[] toArray(List<Double> e) {
        double[] res = new double[e.size()];
        for (int i = 0; i < e.size(); i++) {
            res[i] = e.get(i);
        }
        return res;
    }


    @Deprecated
    public static ArrayRealVector pValue(OLSMultipleLinearRegression rg) {
       return RegressionUtil.pValue(rg);
    }


    @Deprecated
    public static ArrayRealVector yHat(OLSMultipleLinearRegression rg, double[] y) {
       return RegressionUtil.yHat(rg, y);
    }
}
