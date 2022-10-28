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
     * 对多元回归的参数计算t statistics
     * @param rg
     * @return
     */
    public static ArrayRealVector tStatistics(OLSMultipleLinearRegression rg) {
        ArrayRealVector tStatistics = new ArrayRealVector(rg.estimateRegressionParameters())
                .ebeDivide(new ArrayRealVector(rg.estimateRegressionParametersStandardErrors()));

        return tStatistics;
    }

    /**
     * 计算多元回归的pValue
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
     * 求多元回归的y预测值
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
