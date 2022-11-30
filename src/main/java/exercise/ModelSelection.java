package exercise;

import data.Hitters;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.junit.Test;
import tool.modelselection.BestSubsetSelection;
import tool.model.LinearRegressionModel;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/29
 */
public class ModelSelection {

    Hitters hitters = new Hitters();

    /**
     * 6.5.1 best subset selection
     * for my computer it cost 90sec
     */
    @Test
    public void bestSubsetSelection() {
        BestSubsetSelection selection = new BestSubsetSelection(hitters.getX(), hitters.getY(), new LinearRegressionModel(), 8);
        System.out.println(selection.getRes());
    }

    @Test
    public void trainHittersWith6Predictors() {
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(hitters.getX());
        RealMatrix subMatrix = matrix.getSubMatrix(IntStream.range(0, hitters.getY().length).toArray(), new int[]{0, 1, 5, 11, 14, 15});
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(hitters.getY(), subMatrix.getData());
        System.out.println(Arrays.toString(rg.estimateRegressionParameters()));
        // for division i use opposite sign to code, so the efficient is positive.

    }




}
