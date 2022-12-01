package exercise;

import data.Credit;
import data.Hitters;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.junit.Test;
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
     */
    @Test
    public void select11Predictor() {
        BestSubsetSelection selection = new BestSubsetSelection(hitters.getX(), hitters.getY(), new LinearRegressionModel(), null);
        System.out.println(selection.getCacheScoreTrainedByAllData()); // with 11 predictors like textbook
        System.out.println(selection.chooseKWithCv()); // it


    }

}
