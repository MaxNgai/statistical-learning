package exercise;

import data.Auto;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import tool.CrossValidation;
import tool.Macro;

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

        Array2DRowRealMatrix linearTrainXMatrix = Macro.hstack(trainX.toArray());
        Array2DRowRealMatrix linearTestMatrix = Macro.hstack(testX.toArray());
        seeMse(trainY, linearTrainXMatrix, testY, linearTestMatrix);

        Array2DRowRealMatrix quadraticTrainXMatrix = Macro.hstack(trainX.ebeMultiply(trainX).toArray(), trainX.toArray());
        Array2DRowRealMatrix quadraticTestXMatrix = Macro.hstack(testX.ebeMultiply(testX).toArray(), testX.toArray());
        seeMse(trainY, quadraticTrainXMatrix, testY, quadraticTestXMatrix);

        Array2DRowRealMatrix cubicTrainXMatrix = Macro.hstack(trainX.ebeMultiply(trainX).ebeMultiply(trainX).toArray(), trainX.ebeMultiply(trainX).toArray(), trainX.toArray());
        Array2DRowRealMatrix cubicTestXMatrix = Macro.hstack(testX.ebeMultiply(testX).ebeMultiply(testX).toArray(), testX.ebeMultiply(testX).toArray(), testX.toArray());
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
