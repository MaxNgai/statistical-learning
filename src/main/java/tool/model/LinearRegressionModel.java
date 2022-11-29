package tool.model;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import tool.Macro;

import javax.crypto.Mac;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Max Ngai
 * @since 2022/11/30
 */
public class LinearRegressionModel implements Model {

    private RealVector params;
    private double trainRss;

    @Override
    public Model train(RealMatrix x, RealVector y) {
        LinearRegressionModel m = new LinearRegressionModel();
        OLSMultipleLinearRegression rg = new OLSMultipleLinearRegression();
        rg.newSampleData(y.toArray(), x.getData());
        m.params = new ArrayRealVector(rg.estimateRegressionParameters());
        m.trainRss = rg.calculateResidualSumOfSquares();
        return m;
    }

    @Override
    public double predict(RealVector predictors) {
        ArrayRealVector r = new ArrayRealVector();
        RealVector x = r.append(1D).append(predictors);
        return params.dotProduct(x);
    }

    @Override
    public double trainRss() {
        return trainRss;
    }

    @Override
    public double testRss(RealMatrix x, RealVector y) {
        List<double[]> raw = new ArrayList<>();
        for (int i = 0; i < x.getRowDimension(); i++) {
            ArrayRealVector newX = new ArrayRealVector();
            double[] row = newX.append(1D).append(x.getRowVector(i)).toArray();
            raw.add(row);
        }
        RealVector dev = Macro.vstack(raw).transpose().preMultiply(params).subtract(y);
        return dev.dotProduct(dev);
    }
}
