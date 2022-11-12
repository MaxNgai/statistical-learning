package exercise;

import graph.ScatterPlot;
import data.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.RegressionResults;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;
import tool.DefaultVectorChangingVisitor;
import tool.Macro;
import tool.Norm;
import tool.RegressionUtil;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/10/9
 */
public class Regression {

    Advertising advertising = new Advertising();

    Auto auto = new Auto();

    Credit credit = new Credit();

    CarSeat carSeat = new CarSeat();

    Boston boston = new Boston();


    /**
     * p68
     */
    @Test
    public void simpleRegression() {
        SimpleRegression sg = new SimpleRegression();

        double[] tv = advertising.getTV();
        double[] sales = advertising.getY();

        for (int i = 0; i < tv.length; i++) {
            sg.addData(tv[i], sales[i]);
        }

        System.out.println(sg.getIntercept()); // b
        System.out.println(sg.getSlope()); // k
        System.out.println(sg.getInterceptStdErr()); // b's standard error
        System.out.println(sg.getSlopeStdErr()); // k's standard error
        long n = sg.getN(); // sample amount
        double meanSquareError = sg.getMeanSquareError();
        System.out.println(meanSquareError); // mse
        System.out.println(sg.getRSquare()); // R square
        System.out.println(sg.getSignificance()); // p value
        System.out.println(sg.getSumSquaredErrors()); // RSS, residual sum of square
        System.out.println(FastMath.sqrt(sg.getSumSquaredErrors() / (sg.getN() - 2))); // RSE, residual standard error

        System.out.println(sg.getSlope() - sg.getSlopeConfidenceInterval()); // k's 95% confidence interval lower
        System.out.println(sg.getSlope() + sg.getSlopeConfidenceInterval()); // k's 95% confidence interval upper
    }

    /**
     * p74
     */
    @Test
    public void multiRegression() {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(advertising.getY(), advertising.getXMatrix());

        System.out.println(Arrays.toString(rg.estimateRegressionParameters())); // k1, k2, k3, b0

        System.out.println(Arrays.toString(rg.estimateRegressionParametersStandardErrors())); // coefficient std error

        System.out.println(rg.calculateRSquared()); // R square

        System.out.println(((rg.calculateTotalSumOfSquares() - rg.calculateResidualSumOfSquares()) / 3)
                / (rg.calculateResidualSumOfSquares() / (advertising.getAdvertising().getRowDimension() - 3 - 1))); // F-statistics

        System.out.println(rg.estimateRegressionStandardError()); // rse

        System.out.println(rg.calculateResidualSumOfSquares()); // rss

    }

    /**
     * p88
     */
    @Test
    public void interactionMultipleRegression() {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(advertising.getY(), advertising.getTvXRadio());

        System.out.println(Arrays.toString(rg.estimateRegressionParameters())); // b0, k1, k2, k3,
        System.out.println(Arrays.toString(rg.estimateRegressionParametersStandardErrors())); // coefficient std error

        // t-statistics
        ArrayRealVector tStatistics = new ArrayRealVector(rg.estimateRegressionParameters()).ebeDivide(new ArrayRealVector(rg.estimateRegressionParametersStandardErrors()));
        System.out.println(tStatistics);

        // p value
        ArrayRealVector pValue = RegressionUtil.pValue(rg);
        System.out.println(pValue);

    }

    /**
     * p93
     */
    @Test
    public void residualScatterPlot() {
        SimpleRegression rg = new SimpleRegression();
        double[] horsePower = auto.getHorsePower();
        double[] mpg = auto.getMpg();

        for (int i = 0; i < mpg.length; i++) {
            rg.addData(horsePower[i], mpg[i]);
        }

        RegressionResults regress = rg.regress();

        RealVector x = auto.getAuto().getColumnVector(3);
        RealVector yHat = x.mapMultiply(rg.getSlope()).mapAdd(rg.getIntercept());
        RealVector y = auto.getAuto().getColumnVector(0);
        RealVector residual = y.subtract(yHat);
        ScatterPlot.see(yHat.toArray(), residual.toArray());

    }

    /**
     * p102
     */
    @Test
    public void collinearity() {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(credit.getRating(), Macro.hstack(credit.getAge(), credit.getLimit()).getData());


        System.out.println(1 / (1 - rg.calculateRSquared())); // vif
    }

    /**
     * p92
     */
    @Test
    public void polynomialRegression() {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        ArrayRealVector horsepower = new ArrayRealVector(auto.getHorsePower());
        rg.newSampleData(auto.getMpg(), Macro.hstack(
                horsepower.getDataRef(),
                horsepower.ebeMultiply(horsepower).getDataRef()
        ).getData());

        System.out.println(Arrays.toString(rg.estimateRegressionParameters())); // k
        System.out.println(Arrays.toString(rg.estimateRegressionParametersStandardErrors())); // st e


        // t-statistics
        ArrayRealVector x = RegressionUtil.tStatistics(rg);
        System.out.println(x);


        // p value
        ArrayRealVector pValue = RegressionUtil.pValue(rg);
        System.out.println(pValue);
    }

    /**
     * p121-8
     */
    @Test
    public void simpleRegressionOnAuto() {
        SimpleRegression rg = new SimpleRegression();
        double[] y = auto.getMpg();
        double[] x = auto.getHorsePower();

        for (int i = 0; i < y.length; i++) {

            rg.addData(x[i], y[i]);
        }

        System.out.println(rg.getSignificance());  // 有关系
        System.out.println(rg.getSlope()); // how string is the relations,negative
        System.out.println(rg.predict(98)); // predict
        System.out.println((rg.getSlope() + rg.getSlopeConfidenceInterval()) * 98 + rg.getIntercept()); // confidence interval upper
        System.out.println((rg.getSlope() - rg.getSlopeConfidenceInterval()) * 98 + rg.getIntercept()); // confidence interval lower
    }

    /**
     * p121-8
     */
    @Test
    public void multiRegressionOnAuto() {
        List<double[]> collect = IntStream.range(1, 8).boxed().map(e -> auto.getAuto().getColumnVector(e).toArray())
                .collect(Collectors.toList());

        Array2DRowRealMatrix x = Macro.hstack(collect);

        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        double[] y = auto.getMpg();
        rg.newSampleData(y, x.getData());


        System.out.println(((rg.calculateTotalSumOfSquares() - rg.calculateResidualSumOfSquares()) / auto.getAuto().getColumnDimension())
                / (rg.calculateResidualSumOfSquares() / (auto.getAuto().getRowDimension() - auto.getAuto().getColumnDimension() - 1))); // F-statistics


        System.out.println(Arrays.toString(rg.estimateRegressionParameters()));


        // t-statistics
        ArrayRealVector tStatistics = RegressionUtil.tStatistics(rg);
        System.out.println(tStatistics);


        // p value
        ArrayRealVector pValue = RegressionUtil.pValue(rg);
        System.out.println(pValue); // 1,3,5,7 column

        // 残差图
        double[] residuals = rg.estimateResiduals();
        for (int i = 0; i < residuals.length; i++) {
            residuals[i] = Math.abs(residuals[i]);
        }
        ScatterPlot.see(residuals, y);
    }

    /**
     * p101
     */
    @Test
    public void seeIfMyPvalueFunctionCorrect() {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(credit.getBalance(), Macro.hstack(
                credit.getRating(),
                credit.getLimit()
        ).getData());

        System.out.println(RegressionUtil.tStatistics(rg));
        System.out.println(RegressionUtil.pValue(rg));
    }

    /**
     * p123-10
     */
    @Test
    public void carSeatMultipleRegression() {
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(carSeat.getSales(), Macro.hstack(
                carSeat.getPrice(),
                carSeat.getUrban(),
                carSeat.getUS()
        ).getData());

        System.out.println(Arrays.toString(rg.estimateRegressionParameters())); // k

        // p value
        System.out.println(RegressionUtil.pValue(rg)); // urban没用，其他有用
        System.out.println(rg.estimateRegressionStandardError()); // rse
        System.out.println(rg.calculateRSquared()); // rsquare

        OLSMultipleLinearRegression rg2 = new OLSMultipleLinearRegression();
        Array2DRowRealMatrix x = Macro.hstack(
                carSeat.getPrice(),
                carSeat.getUS()
        );
        rg2.newSampleData(carSeat.getSales(), x.getData());
        System.out.println(rg2.estimateRegressionStandardError()); // rse
        System.out.println(rg2.calculateRSquared()); // rsquare


        ArrayRealVector coefficientSte = new ArrayRealVector(rg2.estimateRegressionParametersStandardErrors());
        ArrayRealVector coefficient = new ArrayRealVector(rg2.estimateRegressionParameters());
        System.out.println(coefficient.add(coefficientSte.mapMultiply(2D))); // 95% coefficient interval upper
        System.out.println(coefficient.subtract(coefficientSte.mapMultiply(2D))); // 95% coefficient interval upper

        // 残差图
        ArrayRealVector yHat = RegressionUtil.yHat(rg2, carSeat.getSales());
        ScatterPlot.see(yHat.getDataRef(), rg2.estimateResiduals()); // 没有异常点

        double[] one = new double[carSeat.getSales().length];
        for (int i = 0; i < one.length; i++) {
            one[i] = 1D;
        }

        Array2DRowRealMatrix matrix = Macro.hstack(one, carSeat.getPrice(), carSeat.getUS());
        RealVector operate = matrix.operate(coefficient); // 点积， 这里等于yHat

    }

    /**
     * p121-11
     */
    @Test
    public void noIntercept() {
        GaussianRandomGenerator g = new GaussianRandomGenerator(new JDKRandomGenerator(1));
        int count = 100;
        double[] x = new double[count];
        double[] b = new double[count];
        double[] y = new double[count];
        for (int i = 0; i < count; i++) {
            x[i] = g.nextNormalizedDouble();
        }
        for (int i = 0; i < count; i++) {
            b[i] = g.nextNormalizedDouble();
        }
        for (int i = 0; i < count; i++) {
            y[i] = 2D * x[i] + b[i];
        }

        SimpleRegression rg = new SimpleRegression(false);
        for (int i = 0; i < count; i++) {
            rg.addData(x[i], y[i]);
        }

        System.out.println(rg.getSlope()); // k
        System.out.println(rg.getSignificance()); // p value

        SimpleRegression rg2 = new SimpleRegression();
        for (int i = 0; i < count; i++) {
            rg2.addData(x[i], y[i]);
        }

        System.out.println(rg2.getSlope()); // k 有无截距都差不多
        System.out.println(rg2.getSignificance()); // p value
    }

    /**
     * p124-13
     */
    @Test
    public void testDifferenceVarianceDataInSimpleRg() {

        GaussianRandomGenerator g = new GaussianRandomGenerator(new JDKRandomGenerator(1));
        double var = 0.25D;
        int count = 100;
        double[] x = new double[count];
        double[] xSquare = new double[count];
        double[] exp = new double[count];
        double[] y = new double[count];
        for (int i = 0; i < count; i++) {
            x[i] = g.nextNormalizedDouble();
            xSquare[i] = x[i] * x[i];
        }
        for (int i = 0; i < count; i++) {
            exp[i] = g.nextNormalizedDouble() / (1/var);  // 方差越大, k和b越偏离实际
        }
        for (int i = 0; i < count; i++) {
            y[i] = 0.5D * x[i] + exp[i] - 1;
        }

        ScatterPlot.see(x, y);

        SimpleRegression rg = new SimpleRegression();
        for (int i = 0; i < count; i++) {
            rg.addData(x[i], y[i]);
        }
        System.out.println(rg.getSlope()); // k
        System.out.println(rg.getIntercept()); // b
        System.out.println(rg.getSlopeConfidenceInterval()); // confidence interval, 方差越大，置信区间越宽
        System.out.println(rg.getRSquare()); // R square
        System.out.println(FastMath.sqrt(rg.getSumSquaredErrors() / (rg.getN() - 2))); // RSE, residual standard error

        OLSMultipleLinearRegression mrg = new OLSMultipleLinearRegression();
        mrg.newSampleData(y, Macro.hstack(x, xSquare).getData());
        System.out.println(Arrays.toString(mrg.estimateRegressionParameters())); // coefficients
        System.out.println(RegressionUtil.pValue(mrg)); // p value
        System.out.println(mrg.calculateRSquared()); // r square
        System.out.println(mrg.estimateRegressionStandardError()); // rse
        // 用了多项式效果R-square好一点，但是平方项的置信度不高. RSE反而升高，所以更差了
    }

    /**
     * p125-14
     */
    @Test
    public void collinearityProblem() {
        Random r = new Random();
        GaussianRandomGenerator g = new GaussianRandomGenerator(new JDKRandomGenerator(1));
        int count = 100;
        ArrayRealVector x1 = new ArrayRealVector(Norm.runIf(count));
        RealVector x2 = x1.mapMultiply(0.5D);
        x2.walkInDefaultOrder(new DefaultVectorChangingVisitor(e -> e + g.nextNormalizedDouble() / 10));

        RealVector x2pie = x2.mapMultiply(0.3D);
        x2pie.walkInDefaultOrder(new DefaultVectorChangingVisitor(e -> e + g.nextNormalizedDouble()));
        RealVector x1pie = x1.mapMultiply(2D).mapAdd(2);
        RealVector y = x1pie.add(x2pie);

        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(y.toArray(), Macro.hstack(x1.getDataRef(), x2.toArray()).getData());

        System.out.println(Arrays.toString(rg.estimateRegressionParameters())); // b, k1, k2
//        ScatterPlot.see(x1.getDataRef(), x2.toArray());

        System.out.println(RegressionUtil.pValue(rg));


        SimpleRegression simpleX1 = new SimpleRegression();
        simpleX1.addData(Macro.hstack(x1.getDataRef(), y.toArray()).getData());
        System.out.println(simpleX1.getSignificance());

        SimpleRegression simpleX2 = new SimpleRegression();
        simpleX2.addData(Macro.hstack(x2.toArray(), y.toArray()).getData());
        System.out.println(simpleX2.getSignificance());

        RealVector newX1 = x1.append(0.1);
        RealVector newX2 = x2.append(0.8);
        RealVector newY = y.append(6);
        ScatterPlot.see(newX1.toArray(), newX2.toArray());
        ScatterPlot.see(newX1.toArray(), newX2.toArray(), newY.toArray());


// 怎么在多元回归中找出高杠杆点？
    }

    /**
     * p140-15
     */
    @Test
    public void boston() {
        int nColumn = boston.getX()[0].length;
        Array2DRowRealMatrix x = new Array2DRowRealMatrix(boston.getX());
        double[] y = boston.getCrime();
        for (int i = 0; i < nColumn; i++) {
            SimpleRegression rg = new SimpleRegression();
            double[] column = x.getColumn(i);

            for (int j = 0; j < column.length; j++) {
                rg.addData(column[j], y[j]);
            }

            System.out.println(i + " : " + rg.getSignificance());
//            ScatterPlot.see(column, y); //  感觉都没有线性啊

        }

        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(y, x.getData());
        RealVector pvalue = RegressionUtil.pValue(rg).getSubVector(1, 12);
        System.out.println(pvalue); // p value
        for (int i = 0; i < pvalue.toArray().length; i++) {
            if (pvalue.getEntry(i) <0.01) {
                System.out.println(i); // dis, rad, medv 的参数比较显著
            }
        }

        //  对比多元和简单，很多列在简单中是显著的，在多元不是


        for (int i = 0; i < nColumn; i++) {
            double[] column = x.getColumn(i);
            ArrayRealVector z = new ArrayRealVector(column);
            ArrayRealVector z2 = z.ebeMultiply(z);
            ArrayRealVector z3 = z2.ebeMultiply(z);
            OLSMultipleLinearRegression poly = new OLSMultipleLinearRegression();
            poly.newSampleData(y, Macro.hstack(z.getDataRef(), z2.getDataRef(), z3.getDataRef()).getData());

            try {

                System.out.println(i + " : " + RegressionUtil.pValue(poly));
            } catch (Exception e) {
                //
            }

        }
    }

}

