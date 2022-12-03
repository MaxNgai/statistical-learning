package exercise;

import data.Credit;
import data.Hitters;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.junit.Test;
import tool.Norm;
import tool.RegressionUtil;
import tool.modelselection.BackwardSelection;
import tool.modelselection.BestSubsetSelection;
import tool.model.LinearRegressionModel;
import tool.modelselection.ForwardSelection;
import tool.modelselection.Score;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/29
 */
public class LinearModelSelection {

    Hitters hitters = new Hitters();

    Credit credit = new Credit();


    /**
     * 6.5.1 best subset selection
     * for my computer it cost 90sec
     */
    @Test
    public void bestSubsetSelection() {
        BestSubsetSelection selection = new BestSubsetSelection(hitters.getX(), hitters.getY(), new LinearRegressionModel(), 8);
        System.out.println(selection.getCacheScoreTrainedByAllData());
    }

    /**
     * p247
     */
    @Test
    public void trainHittersWith6Predictors() {
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(hitters.getX());
        RealMatrix subMatrix = matrix.getSubMatrix(IntStream.range(0, hitters.getY().length).toArray(), new int[]{0, 1, 5, 11, 14, 15});
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(hitters.getY(), subMatrix.getData());
        System.out.println(Arrays.toString(rg.estimateRegressionParameters()));
        // for division i use opposite sign to code, so the efficient is positive.

    }

    /**
     * p247
     */
    @Test
    public void forwardSelection() {
        ForwardSelection fs = new ForwardSelection(hitters.getX(), hitters.getY(), new LinearRegressionModel(), 7);
        System.out.println(fs.getCacheScoreTrainedByAllData());
        fs.getCacheScoreTrainedByAllData().stream().filter(e -> e.getSelectedX().getSize() == 7)
                .forEach(e -> System.out.println(e.getModel()));

    }


    /**
     * p247
     */
    @Test
    public void backwardSelection() {
        BackwardSelection fs = new BackwardSelection(hitters.getX(), hitters.getY(), new LinearRegressionModel(), 7);
        System.out.println(fs.getCacheScoreTrainedByAllData());

    }

    /**
     * p209
     */
    @Test
    public void bestSubsetAndForwardSelectionOnCredit() {
        BestSubsetSelection b = new BestSubsetSelection(credit.getX().getData(), credit.getY().toArray(), new LinearRegressionModel(), 4);
        System.out.println(b.getCacheScoreTrainedByAllData());
        System.out.println();
        ForwardSelection f = new ForwardSelection(credit.getX().getData(), credit.getY().toArray(), new LinearRegressionModel(), 4);
        System.out.println(f.getCacheScoreTrainedByAllData());
    }

    /**
     * p250
     * it takes a few minutes to run
     */
    @Test
    public void select11Predictor() {
        BestSubsetSelection selection = new BestSubsetSelection(hitters.getX(), hitters.getY(), new LinearRegressionModel(), null);
        System.out.println(selection.getCacheScoreTrainedByAllData()); // with 11 predictors like textbook, not easy...
        System.out.println(selection.chooseKWithCv()); // it


    }

    /**
     * applied
     * p262-8
     */
    @Test
    public void polynomialSelection() {
        int n = 100;
        double[] x = Norm.rnorm(n);
        ArrayRealVector e = new ArrayRealVector(Norm.rnorm(n));
        Array2DRowRealMatrix polynomial = RegressionUtil.polynomial(x, 3);
        ArrayRealVector k = new ArrayRealVector(new double[]{3, 2, 1});
        ArrayRealVector kx = new ArrayRealVector(polynomial.transpose().preMultiply(k.getDataRef()));
        ArrayRealVector Y = kx.add(e).add(new ArrayRealVector(n, 4)); // y = e + 4 + 3x + 2x^2 + x^3
        Array2DRowRealMatrix X = RegressionUtil.polynomial(x, 10);

        BestSubsetSelection best = new BestSubsetSelection(X.getData(), Y.toArray(), new LinearRegressionModel(), null);
        System.out.println(best.getCacheScoreTrainedByAllData());
        System.out.println(best.chooseKWithCv()); // 3

        ForwardSelection forward = new ForwardSelection(X.getData(), Y.toArray(), new LinearRegressionModel(), null);
        System.out.println(forward.getCacheScoreTrainedByAllData());
        System.out.println(forward.chooseKWithCv()); // 3

        BackwardSelection back = new BackwardSelection(X.getData(), Y.toArray(), new LinearRegressionModel(), null);
        System.out.println(back.getCacheScoreTrainedByAllData());
        System.out.println(back.chooseKWithCv()); // 4

    }

    /**
     * p262-8-f
     */
    @Test
    public void sevenPower() {
        int n = 100;
        double[] x = Norm.rnorm(n);
        ArrayRealVector e = new ArrayRealVector(Norm.rnorm(n));
        Array2DRowRealMatrix seven = RegressionUtil.polynomial(x, 7);
        RealVector Y = seven.getColumnVector(6).mapMultiply(7).add(e).mapAdd(1); // y = 7x^7 + e + 1
        Array2DRowRealMatrix X = RegressionUtil.polynomial(x, 10);

        BestSubsetSelection best = new BestSubsetSelection(X.getData(), Y.toArray(), new LinearRegressionModel(), null);
        System.out.println(best.getCacheScoreTrainedByAllData());
        System.out.println(best.chooseKWithCv()); // 1, which is the x^7
    }
}
