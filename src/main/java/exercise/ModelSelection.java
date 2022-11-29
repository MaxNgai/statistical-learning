package exercise;

import data.Hitters;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.junit.Test;
import tool.BestSubsetSelection;
import tool.model.LinearRegressionModel;

/**
 * @author Max Ngai
 * @since 2022/11/29
 */
public class ModelSelection {

    Hitters hitters = new Hitters();

    /**
     * 6.5.1 best subset selection
     */
    @Test
    public void bestSubsetSelection() {
        BestSubsetSelection selection = new BestSubsetSelection(hitters.getX(), hitters.getY(), new LinearRegressionModel(), 8);
        System.out.println(selection.getBestPredictorSet());
    }


}
