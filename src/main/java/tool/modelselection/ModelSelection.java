package tool.modelselection;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import tool.model.Model;

import java.util.List;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/30
 */
public abstract class ModelSelection {
    RealMatrix X;
    RealVector Y;
    int n;
    int p;
    Model model;
    Integer maxPredictors;

    int[] rowsIndex;
    List<Score> res;
    final int kFold = 10;

    ModelSelection(double[][] X, double[] Y, Model model, Integer maxPredictors) {
        this.X = new Array2DRowRealMatrix(X);
        this.Y = new ArrayRealVector(Y);
        this.model = model;
        n = X.length;
        p = X[0].length;
        rowsIndex = IntStream.range(0, n).toArray();
        this.maxPredictors = maxPredictors == null ? p : Math.min(p, maxPredictors);
    }

    public List<Score> getRes() {
        return res;
    }

    public RealMatrix getXByPredictors(PredictorCombo combo) {
        int[] columns = combo.getArray();
        RealMatrix subMatrix = X.getSubMatrix(rowsIndex, columns);
        return subMatrix;
    }

}
