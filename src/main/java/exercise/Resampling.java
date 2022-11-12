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

        Pair<RealMatrix, RealMatrix> raw = CrossValidation.validationSet(auto.getAuto());
        RealMatrix train = raw.getFirst();
        RealMatrix test = raw.getSecond();
        RealVector testX = test.getColumnVector(3);
        RealVector testY = test.getColumnVector(0);
        RealVector trainX = train.getColumnVector(3);
        RealVector trainY = train.getColumnVector(0);

        Array2DRowRealMatrix linearTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 1);
        Array2DRowRealMatrix linearTestMatrix = RegressionUtil.polynomial(testX.toArray(), 1);
        seeRegressionMse(trainY, linearTrainXMatrix, testY, linearTestMatrix);

        Array2DRowRealMatrix quadraticTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 2);
        Array2DRowRealMatrix quadraticTestXMatrix = RegressionUtil.polynomial(testX.toArray(), 2);
        seeRegressionMse(trainY, quadraticTrainXMatrix, testY, quadraticTestXMatrix);

        Array2DRowRealMatrix cubicTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 3);
        Array2DRowRealMatrix cubicTestXMatrix = RegressionUtil.polynomial(testX.toArray(), 3);
        seeRegressionMse(trainY, cubicTrainXMatrix, testY, cubicTestXMatrix);
    }

    private static double seeRegressionMse(RealVector trainY, RealMatrix trainX, RealVector testY, RealMatrix testX) {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(trainY.toArray(), trainX.getData());
        ArrayRealVector param = new ArrayRealVector(rg.estimateRegressionParameters());
        double b = param.getEntry(0);
        RealVector k = param.getSubVector(1, param.getDimension() - 1);

        RealVector dev = testX.transpose().preMultiply(k).add(new ArrayRealVector(testX.getRowDimension(), b)).subtract(testY);
        double mse = dev.dotProduct(dev) / dev.getDimension();
        return mse;

    }

    CrossValidation.CvMseGetter regressionMseGetter = new CrossValidation.CvMseGetter() {
        @Override
        public <TEX extends RealMatrix, TEY extends RealVector, TRX extends RealMatrix, TRY extends RealVector> double testSetMse(TRX trainX, TRY trainY, TEX testX, TEY testY) {
            return seeRegressionMse(trainY, trainX, testY, testX);
        }
    };

    /**
     * p192 loocv
     */
    @Test
    public void loocv() {
        ArrayRealVector y = new ArrayRealVector(auto.getMpg());
        Array2DRowRealMatrix x = Macro.hstack(auto.getHorsePower());

        double avgMse = CrossValidation.loocvMse(x, y, regressionMseGetter);
        System.out.println(avgMse);
    }

    /**
     * p193
     */
    @Test
    public void polynomialLoocv() {
        int power = 5;
        for (int i = 1; i <= power; i++) {
            ArrayRealVector y = new ArrayRealVector(auto.getMpg());
            Array2DRowRealMatrix x = RegressionUtil.polynomial(auto.getHorsePower(), i);
            double avgMse = CrossValidation.loocvMse(x, y, regressionMseGetter);
            System.out.println(avgMse);
        }
    }




}
