package tool.model;

import algo.LDA;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author Max Ngai
 * @since 2022/11/30
 */
public class LdaModel implements Model {

    private LDA lda;
    private double trainRss;

    @Override
    public Model train(RealMatrix x, RealVector y) {
        LdaModel m = new LdaModel();
        LDA lda = new LDA(x.getData(), y.toArray());
        m.lda = lda;
        int F = 0;
        for (int i = 0; i < y.getDimension(); i++) {
            double yHat = m.predict(x.getRowVector(i));
            if (yHat != y.getEntry(i)) {
                F++;
            }
        }
        m.trainRss = F;
        return m;
    }

    @Override
    public double predict(RealVector x) {
        return lda.predict(x.toArray());
    }

    @Override
    public double trainRss() {
        return trainRss;
    }

    @Override
    public double testRss(RealMatrix x, RealVector y) {
        int F = 0;
        for (int i = 0; i < x.getRowDimension(); i++) {
            if (predict(x.getRowVector(i)) != y.getEntry(i)) {
                F++;
            }
        }
        return F;
    }
}
