package exercise;

import algo.LDA;
import com.google.common.collect.Lists;
import data.*;
import graph.ScatterPlot;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import tool.CrossValidation;
import tool.Macro;
import tool.Norm;
import tool.RegressionUtil;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * @author Max Ngai
 * @since 2022/11/12
 */
public class Resampling {
    Auto auto = new Auto();

    Portfolio portfolio = new Portfolio();

    Default aDefault = new Default();

    Weekly weekly = new Weekly();

    Boston boston = new Boston();

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

    /**
     * p198-5-c
     *
     * here we use lda rather than logistics regression as textbook required.
     * they are very similar and we are testing cv, so it should not be a big deal
     */
    @Test
    public void ldaOnDefaultUsingValidationSet() {
        RealVector income = new ArrayRealVector(aDefault.getIncome()).mapDivide(1000);
        Array2DRowRealMatrix XY = Macro.hstack(income.toArray(), aDefault.getBalance(), aDefault.getDefault());

        for (int j = 0; j < 3; j++) {
            Pair<RealMatrix, RealMatrix> split = CrossValidation.validationSet(XY);
            RealMatrix trainMatrix = split.getFirst();
            RealMatrix testMatrix = split.getSecond();
            LDA lda = new LDA(trainMatrix.getSubMatrix(0, trainMatrix.getRowDimension() - 1, 0, 1).getData(), trainMatrix.getColumnVector(2).toArray());

            int F = 0;
            for (int i = 0; i < testMatrix.getRowDimension(); i++) {
                double[] row = testMatrix.getRow(i);
                if (lda.predict(new double[]{row[0], row[1]}) != row[2]) {
                    F++;
                }
            }
            System.out.println(((double) F) / testMatrix.getRowDimension()); // error rate
        }
    }

    /**
     * p198-5-d
     */
    @Test
    public void ldaOnDefaultWithStudentUsingValidationSet() {
        RealVector income = new ArrayRealVector(aDefault.getIncome()).mapDivide(1000);
        Array2DRowRealMatrix XY = Macro.hstack(income.toArray(), aDefault.getBalance(), aDefault.getStudent(), aDefault.getDefault());

        for (int j = 0; j < 3; j++) {
            Pair<RealMatrix, RealMatrix> split = CrossValidation.validationSet(XY);
            RealMatrix trainMatrix = split.getFirst();
            RealMatrix testMatrix = split.getSecond();
            LDA lda = new LDA(trainMatrix.getSubMatrix(0, trainMatrix.getRowDimension() - 1, 0, 2).getData(), trainMatrix.getColumnVector(3).toArray());

            int F = 0;
            for (int i = 0; i < testMatrix.getRowDimension(); i++) {
                double[] row = testMatrix.getRow(i);
                if (lda.predict(new double[]{row[0], row[1], row[2]}) != row[3]) {
                    F++;
                }
            }
            System.out.println(((double) F) / testMatrix.getRowDimension()); // error rate

            /*
             introducing isStudent cannot reduce error rate
             */
        }
    }

    /**
     * p200-7-e
     */
    @Test
    public void loocvOnWeekly() {
        Array2DRowRealMatrix x = Macro.hstack(weekly.getLag1(), weekly.getLag2());
        ArrayRealVector y = new ArrayRealVector(weekly.getDirection());
        LDA lda = new LDA(x.getData(), y.toArray());
        System.out.println(lda.errorRate());

        System.out.println(CrossValidation.loocvMse(x, y, new CrossValidation.CvMseGetter() {
            @Override
            public <TEX extends RealMatrix, TEY extends RealVector, TRX extends RealMatrix, TRY extends RealVector> double testSetMse(TRX trainX, TRY trainY, TEX testX, TEY testY) {
                LDA lda2 = new LDA(trainX.getData(), trainY.toArray());
                int F = 0;
                for (int i = 0; i < testY.getDimension(); i++) {
                    if (lda2.predict(testX.getData()[i]) != testY.toArray()[i]) {
                        F++;
                    }
                }
                return ((double) F) / testY.getDimension();
            }
        }));

        /*
                error rate of loocv is greater. reasonable
         */
    }

    /**
     * p200-8
     */
    @Test
    public void randomGenerateNumberTest() {
        int n = 100;
        ArrayRealVector x = new ArrayRealVector(Norm.rnorm(n));
        ArrayRealVector e = new ArrayRealVector(Norm.rnorm(n));
        RealVector y = x.ebeMultiply(x).mapMultiply(-2).add(x).add(e);
//        ScatterPlot.see(x.getDataRef(), y.toArray());


        for (int power = 1; power <= 4; power++) {
            AtomicInteger power0 = new AtomicInteger(power);
            double mse = CrossValidation.loocvMse(RegressionUtil.polynomial(x.getDataRef(), power), y, new CrossValidation.CvMseGetter() {

                @Override
                public <TEX extends RealMatrix, TEY extends RealVector, TRX extends RealMatrix, TRY extends RealVector> double testSetMse(TRX trainX, TRY trainY, TEX testX, TEY testY) {
                    OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
                    rg.newSampleData(trainY.toArray(), trainX.getData());
//                    System.out.println(RegressionUtil.pValue(rg)); // p-value, when power > 1 p-value of intercept is high because the intercept is norm and mean is zero
                    List<Double> collect = Arrays.stream(rg.estimateRegressionParameters()).boxed().collect(Collectors.toList());
                    Double b = collect.get(0);
                    ArrayRealVector k = new ArrayRealVector(Macro.toArray(collect.subList(1, collect.size())));

                    RealVector dev = testX.transpose().preMultiply(k).mapAdd(b).subtract(testY);

                    return dev.dotProduct(dev) / (n - 1 - power0.get());
                }
            });

            System.out.println(mse); // when power = 2 mse has the smallest value
        }
    }

    /**
     * p201-9
     */
    @Test
    public void boston() {
        SummaryStatistics summary = new SummaryStatistics();
        for (double v : boston.getMedv()) {
            summary.addValue(v);
        }
        System.out.println("mean = " + summary.getMean());
        double stdError = summary.getStandardDeviation() / Math.sqrt(boston.getMedv().length);
        System.out.println("stdError = " + stdError);

        List<Double> stdErrorFromBootstrap = CrossValidation.bootstrapGetSE(1000, Macro.hstack(boston.getMedv()), new ArrayRealVector(boston.getMedv().length), new CrossValidation.ParamGetter() {
            @Override
            public <X extends RealMatrix, Y extends RealVector> List<Double> getParams(X x, Y y) {
                SummaryStatistics s = new SummaryStatistics();
                for (double v : x.getColumn(0)) {
                    s.addValue(v);
                }
                return Collections.singletonList(s.getMean());
            }
        });
        System.out.println("stdErrorFromBootstrap = " + stdErrorFromBootstrap);
        System.out.println("intervalFromBootstrap = " + Arrays.asList(summary.getMean() - 2 * stdErrorFromBootstrap.get(0), summary.getMean() + 2 * stdErrorFromBootstrap.get(0)));
        System.out.println("interval = " + Arrays.asList(summary.getMean() - 2 * stdError, summary.getMean() + 2 * stdError));

        Median median = new Median();
        median.setData(boston.getMedv());
        System.out.println("median = " + median.evaluate());
        List<Double> medianStdErrorFromBootstrap = CrossValidation.bootstrapGetSE(1000, Macro.hstack(boston.getMedv()), new ArrayRealVector(boston.getMedv().length), new CrossValidation.ParamGetter() {
            @Override
            public <X extends RealMatrix, Y extends RealVector> List<Double> getParams(X x, Y y) {
                Median median = new Median();
                median.setData(x.getColumn(0));
                return Collections.singletonList(median.evaluate());
            }
        });
        System.out.println("medianStdErrorFromBootstrap = " + medianStdErrorFromBootstrap);


        Percentile percentile = new Percentile();
        percentile.setQuantile(10D);
        percentile.setData(boston.getMedv());
        System.out.println("10thPercentile = " + percentile.evaluate());
        List<Double> percentileStdErrorFromBootstrap = CrossValidation.bootstrapGetSE(1000, Macro.hstack(boston.getMedv()), new ArrayRealVector(boston.getMedv().length), new CrossValidation.ParamGetter() {
            @Override
            public <X extends RealMatrix, Y extends RealVector> List<Double> getParams(X x, Y y) {
                Percentile percentile = new Percentile();
                percentile.setQuantile(10D);
                percentile.setData(x.getColumn(0));
                return Collections.singletonList(percentile.evaluate());
            }
        });

        System.out.println("percentileStdErrorFromBootstrap = " + percentileStdErrorFromBootstrap);

    }
}
