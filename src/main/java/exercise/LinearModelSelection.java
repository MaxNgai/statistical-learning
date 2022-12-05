package exercise;

import data.Boston;
import data.College;
import data.Credit;
import data.Hitters;
import graph.ScatterPlot;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.junit.Test;
import tool.Macro;
import tool.Norm;
import tool.RegressionUtil;
import tool.model.LdaModel;
import tool.modelselection.BackwardSelection;
import tool.modelselection.BestSubsetSelection;
import tool.model.LinearRegressionModel;
import tool.modelselection.ForwardSelection;
import tool.modelselection.Score;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * @author Max Ngai
 * @since 2022/11/29
 */
public class LinearModelSelection {

    Hitters hitters = new Hitters();

    Credit credit = new Credit();

    College college = new College();

    Boston boston = new Boston();



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

    /**
     * to see in college data, what predictors is truely useful
     */
    @Test
    public void collegeBestSubSet() {
        BestSubsetSelection best = new BestSubsetSelection(college.getX(), college.getY(), new LinearRegressionModel(), null);
        System.out.println(best.getCacheScoreTrainedByAllData());
        System.out.println(best.chooseKWithCv()); // 8

    }

    /**
     * p264-10
     */
    @Test
    public void testAndTrainTrap() {
        int p = 20;
        int n = 1000;
        List<double[]> collect = IntStream.range(0, p + 1).boxed().map(e -> Norm.rnorm(n))
                .collect(Collectors.toList());
        Array2DRowRealMatrix raw = Macro.hstack(collect);
        Random random = new Random();
        double[] rawK = IntStream.range(1, p + 2).mapToDouble(e -> e % 2 != 0 ? (double)random.nextInt(100) : 0D).toArray();
        rawK[rawK.length - 1] = 1;
        ArrayRealVector k = new ArrayRealVector(rawK);
        RealVector y = raw.transpose().preMultiply(k); // y = x1 + 3 * x3 + 5 * x5 ... + 19 * x19 + e
        RealMatrix x = raw.getSubMatrix(0, raw.getRowDimension() - 1, 0, raw.getColumnDimension() - 2);

        RealVector trainY = y.getSubVector(0, 900);
        RealVector testY = y.getSubVector(900, 100);
        RealMatrix trainX = x.getSubMatrix(0, 899, 0, p - 1);
        RealMatrix testX = x.getSubMatrix(900, 999, 0, p - 1);

        BestSubsetSelection best = new BestSubsetSelection(trainX.getData(), trainY.toArray(), new LinearRegressionModel(), null);
        List<Score> score = best.getCacheScoreTrainedByAllData();
        score.stream().sorted(Comparator.comparingDouble(s -> s.getSelectedX().getSize()))
                .forEach(e -> {
                    int[] col = e.getSelectedX().getSelected().stream().mapToInt(b -> b).toArray();
                    int[] rows = IntStream.range(0, testX.getRowDimension()).toArray();
                    double testRss = e.getModel().testRss(testX.getSubMatrix(rows, col), testY);
                    e.setTestRss(testRss);
                });

        double[] xAxis = IntStream.range(1, p + 1).mapToDouble(e -> e).toArray();
        double[] trainRss = score.stream().mapToDouble(e -> e.getTrainRss() / 900D).toArray();
        double[] testRss = score.stream().mapToDouble(e -> e.getTestRss() / 100D).toArray();
        double[] sqrt = score.stream().mapToDouble(e -> {
            ArrayRealVector v = new ArrayRealVector(p, 0D);
            LinearRegressionModel model = (LinearRegressionModel) e.getModel();
            RealVector params = model.getParams();
            List<Integer> selected = e.getSelectedX().getSelected();
            for (int i = 0; i < selected.size(); i++) {
                v.setEntry(selected.get(i), params.getEntry(i));
            }

            RealVector dev = v.subtract(k.getSubVector(0, p));
            return Math.sqrt(dev.dotProduct(dev));
        }).toArray();
        ScatterPlot.see(xAxis, trainRss);
        ScatterPlot.see(xAxis, testRss);
        ScatterPlot.see(xAxis, sqrt);
        System.out.println(score);

        score.sort(Comparator.comparingDouble(e -> e.getTestRss()));
        System.out.println(score.get(0).getSelectedX().getSize() + " is the best model");
        System.out.println(score.get(0)); // # best model has 16 predictors


    }

    /**
     * p264-11
     */
    @Test
    public void boston() {
        BestSubsetSelection best = new BestSubsetSelection(boston.getX(), boston.getCrime(), new LinearRegressionModel(), null);
        System.out.println(best.getCacheScoreTrainedByAllData());
        System.out.println(best.chooseKWithCv());
    }


}
