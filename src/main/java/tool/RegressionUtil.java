package tool;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Max Ngai
 * @since 2022/10/12
 */
public class RegressionUtil {
    /**
     * get t statistics for parameters in Multiple linear regression
     * @param rg
     * @return
     */
    public static ArrayRealVector tStatistics(OLSMultipleLinearRegression rg) {
        ArrayRealVector tStatistics = new ArrayRealVector(rg.estimateRegressionParameters())
                .ebeDivide(new ArrayRealVector(rg.estimateRegressionParametersStandardErrors()));

        return tStatistics;
    }

    /**
     * get p value for parameters in Multiple linear regression
     * @param rg
     * @return
     */
    public static ArrayRealVector pValue(OLSMultipleLinearRegression rg) {
        ArrayRealVector tStatistics = tStatistics(rg);

        // p value
        TDistribution tDistribution = new TDistribution(rg.estimateResiduals().length);
        List<Double> pValue = Arrays.stream(tStatistics.getDataRef())
                .map(e -> 2D * (1D - tDistribution.cumulativeProbability(Math.abs(e))))
                .boxed().collect(Collectors.toList());

        return new ArrayRealVector(Macro.toArray(pValue));
    }

    /**
     * use y to get y hat in for  Multiple linear regression
     * @param rg
     * @param y
     * @return
     */
    public static ArrayRealVector yHat(OLSMultipleLinearRegression rg, double[] y) {
        ArrayRealVector data = new ArrayRealVector(y);
        ArrayRealVector residual = new ArrayRealVector(rg.estimateResiduals());
        ArrayRealVector yHat = data.subtract(residual);
        return yHat;
    }
}
