package exercise;

import com.google.common.collect.Lists;
import data.Auto;
import data.Portfolio;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import tool.CrossValidation;
import tool.Macro;
import tool.RegressionUtil;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Max Ngai
 * @since 2022/11/12
 */
public class Resampling {
    Auto auto = new Auto();

    Portfolio portfolio = new Portfolio();

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
        System.out.println(seeRegressionMse(trainY, linearTrainXMatrix, testY, linearTestMatrix));

        Array2DRowRealMatrix quadraticTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 2);
        Array2DRowRealMatrix quadraticTestXMatrix = RegressionUtil.polynomial(testX.toArray(), 2);
        System.out.println(seeRegressionMse(trainY, quadraticTrainXMatrix, testY, quadraticTestXMatrix));

        Array2DRowRealMatrix cubicTrainXMatrix = RegressionUtil.polynomial(trainX.toArray(), 3);
        Array2DRowRealMatrix cubicTestXMatrix = RegressionUtil.polynomial(testX.toArray(), 3);
        System.out.println(seeRegressionMse(trainY, cubicTrainXMatrix, testY, cubicTestXMatrix));
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

    /**
     * p193
     *
     * if use k = 10 then n % k != 0
     * eventually I make it k+1 parts where size of the last part smaller than n/k
     * but it still works
     */
    @Test
    public void kFoldCv() {
        int power = 5;
        for (int i = 1; i <= power; i++) {
            ArrayRealVector y = new ArrayRealVector(auto.getMpg());
            Array2DRowRealMatrix x = RegressionUtil.polynomial(auto.getHorsePower(), i);
            double avgMse = CrossValidation.kFoldCv(x, y, regressionMseGetter, 14);
            System.out.println(avgMse);
        }
    }

    @Test
    public void portfolioAlphaTest() {
        double alpha = getPortfolioAlpha(portfolio.getData().getData());

        System.out.println(alpha);
    }

    private double getPortfolioAlpha(double[][] xy) {
        Covariance covariance = new Covariance(xy);
        double varX = covariance.getCovarianceMatrix().getEntry(0, 0);
        double varY = covariance.getCovarianceMatrix().getEntry(1, 1);
        double covXY = covariance.getCovarianceMatrix().getEntry(0, 1);

        double alpha = (varY - covXY) / (varX + varY - 2 * covXY);
        return alpha;
    }

    @Test
    public void bootstrap() {

        double[] x = portfolio.getX();
        double[] y = portfolio.getY();
        List<Double> doubles = CrossValidation.bootstrapGetSE(1000, Macro.hstack(x), new ArrayRealVector(y), new CrossValidation.ParamGetter() {
            @Override
            public <X extends RealMatrix, Y extends RealVector> List<Double> getParams(X x, Y y) {
                double[][] xy = Macro.hstack(x.getColumn(0), y.toArray()).getData();
                return Lists.newArrayList(getPortfolioAlpha(xy));
            }
        });

        System.out.println(doubles);

    }

    /**
     * p196
     */
    @Test
    public void horsepowerRegressionBootstrap() {
        List<Double> doubles = CrossValidation.bootstrapGetSE(1000, Macro.hstack(auto.getHorsePower()), new ArrayRealVector(auto.getMpg()), new CrossValidation.ParamGetter() {
            @Override
            public <X extends RealMatrix, Y extends RealVector> List<Double> getParams(X x, Y y) {
                SimpleRegression rg = new SimpleRegression();
                Array2DRowRealMatrix hstack = Macro.hstack(x.getColumnVector(0).toArray(), y.toArray());
                rg.addData(hstack.getData());
                return Lists.newArrayList(rg.getIntercept(), rg.getSlope());
            }
        });

        System.out.println(doubles);

        SimpleRegression rg = new SimpleRegression();
        rg.addData(Macro.hstack(auto.getHorsePower(), auto.getMpg()).getData());
        System.out.println(Arrays.asList(rg.getInterceptStdErr(), rg.getSlopeStdErr()));

        /**
         * original regression api 's std. error is different from bootstrap's.
         * the bootstrap is correct.
         * simple regression assumes xy is linear however now it is not.
         */
    }

    /**
     * p197
     */
    @Test
    public void quadraticRegressionBootstrap() {
        Array2DRowRealMatrix polynomial = RegressionUtil.polynomial(auto.getHorsePower(), 2);
        List<Double> doubles = CrossValidation.bootstrapGetSE(1000, polynomial, new ArrayRealVector(auto.getMpg()), new CrossValidation.ParamGetter() {
            @Override
            public <X extends RealMatrix, Y extends RealVector> List<Double> getParams(X x, Y y) {
                OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
                rg.newSampleData(y.toArray(), x.getData());
                return Arrays.stream(rg.estimateRegressionParameters()).boxed().collect(Collectors.toList());
            }
        });

        System.out.println(doubles);

        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(auto.getMpg(), polynomial.getData());
        System.out.println(Arrays.stream(rg.estimateRegressionParametersStandardErrors()).boxed().collect(Collectors.toList()));
    }
}
