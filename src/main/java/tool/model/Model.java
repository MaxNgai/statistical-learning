package tool.model;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * define a model that should yield performance score.
 * please invoke {@link #train(RealMatrix, RealVector)} first
 *
 * @author Max Ngai
 * @since 2022/11/29
 */
public interface Model {

    /**
     * first method to invoke before other method.
     * each time of train will return a new model.
     *  thread safe.
     * @param x
     * @param y
     * @implNote make sure thread safe by returning new model
     */
    Model train(RealMatrix x, RealVector y);

    double predict(RealVector x);

    /**
     * because trained, so no need param
     * @return
     */
    double trainRss();

    /**
     *
     * @param x X in testSet
     * @param y Y in testSet
     * @return
     */
    double testRss(RealMatrix x, RealVector y);

    default double testMse(RealMatrix x, RealVector y) {
        return testRss(x, y) / x.getRowDimension();
    }




}
