package exercise;

import data.Auto;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import tool.CrossValidation;
import tool.Macro;
import tool.RegressionUtil;

/**
 * @author Max Ngai
 * @since 2022/11/12
 */
public class Resampling {
    Auto auto = new Auto();


    /**
     * p191
     */
    @Test
    public void validationSet() {

        Pair<RealMatrix, RealMatrix> raw = CrossValidation.validationSet(auto.getAuto(), 196);
        RealMatrix train = raw.getFirst();
        RealMatrix test = raw.getSecond();
        RealVector testX = test.getColumnVector(3);
        RealVector testY = test.getColumnVector(0);
        RealVector trainX = train.getColumnVector(3);
        RealVector trainY = train.getColumnVector(0);

        Array2DRowRealMatrix linearTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 1);
        Array2DRowRealMatrix linearTestMatrix = RegressionUtil.polynomial(testX.toArray(), 1);
        seeMse(trainY, linearTrainXMatrix, testY, linearTestMatrix);

        Array2DRowRealMatrix quadraticTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 2);
        Array2DRowRealMatrix quadraticTestXMatrix = RegressionUtil.polynomial(testX.toArray(), 2);
        seeMse(trainY, quadraticTrainXMatrix, testY, quadraticTestXMatrix);

        Array2DRowRealMatrix cubicTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 3);
        Array2DRowRealMatrix cubicTestXMatrix = RegressionUtil.polynomial(testX.toArray(), 3);
        seeMse(trainY, cubicTrainXMatrix, testY, cubicTestXMatrix);
    }

    private static double seeMse(RealVector trainY, RealMatrix trainX, RealVector testY, RealMatrix testX) {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(trainY.toArray(), trainX.getData());
        ArrayRealVector param = new ArrayRealVector(rg.estimateRegressionParameters());
        double b = param.getEntry(0);
        RealVector k = param.getSubVector(1, param.getDimension() - 1);

        RealVector dev = testX.transpose().preMultiply(k).add(new ArrayRealVector(testX.getRowDimension(), b)).subtract(testY);
        double mse = dev.dotProduct(dev) / dev.getDimension();
        System.out.println(mse);
        return mse;

    }

}
