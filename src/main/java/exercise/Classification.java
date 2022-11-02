package exercise;

import data.Caravan;
import data.Default;
import data.Smarket;
import org.apache.commons.math3.analysis.function.Logistic;
import org.apache.commons.math3.analysis.function.Logit;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.distribution.LogisticDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.junit.Test;
import tool.*;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * @author Max Ngai
 * @since 2022/10/17
 */
public class Classification {

    Default aDefault = new Default();

    Smarket smarket = new Smarket();

    Caravan caravan = new Caravan();

    /**
     * p145
     */
    @Test
    public void lda() {
        System.out.println(Covariance.covForLda(aDefault.getStudentAndBalance(), aDefault.getDefault()));

        LDA lda = new LDA(aDefault.getStudentAndBalance(), aDefault.getDefault());

        System.out.println(lda.getConfusionMatrix());
        System.out.println(Arrays.stream(lda.getYHat()).sum());
    }

    /**
     * p146
     * p > 0.2
     */
    @Test
    public void ldaWithThreshold() {
        // unable to change threshold
    }

    /**
     * p149
     */
    @Test
    public void covarianceOfQda() {
        System.out.println(Covariance.covForQda(aDefault.getStudentAndBalance(), aDefault.getDefault()));
    }

    /**
     * qda on 'default' dataset, not in text but validated by python sklearn
     */
    @Test
    public void qda() {
        QDA qda = new QDA(aDefault.getStudentAndBalance(), aDefault.getDefault());
        System.out.println(qda.getConfusionMatrix());
    }

    /**
     * p155
     */
    @Test
    public void corForSmarket() {
        org.apache.commons.math3.stat.correlation.Covariance cov =
                new org.apache.commons.math3.stat.correlation.Covariance(smarket.getBesidesDirection());
        PearsonsCorrelation cor = new PearsonsCorrelation(smarket.getBesidesDirection());
        System.out.println(cor.getCorrelationMatrix());
    }

    /**
     * p161
     */
    @Test
    public void ldaForsmarket() {
        Array2DRowRealMatrix x = Macro.matrixHConcat(smarket.getLag1(), smarket.getLag2());
        RealMatrix trainX = x.getSubMatrix(0, 997, 0, 1);
        RealMatrix testX = x.getSubMatrix(998, 1249, 0, 1);
        ArrayRealVector y = new ArrayRealVector(smarket.getDirection());
        RealVector trainY = y.getSubVector(0, 998);
        RealVector testY = y.getSubVector(998, 252);

        LDA model = new LDA(trainX.getData(), trainY.toArray());
        ArrayRealVector yHat = new ArrayRealVector(Macro.toArray(Arrays.stream(testX.getData()).map(model::predict).collect(Collectors.toList())));
        ConfusionMatrix confusion = new ConfusionMatrix(yHat.getDataRef(), testY.toArray());
        System.out.println(confusion);

    }

    /**
     * p163
     */
    @Test
    public void qdaForsmarket() {
        Array2DRowRealMatrix x = Macro.matrixHConcat(smarket.getLag1(), smarket.getLag2());
        RealMatrix trainX = x.getSubMatrix(0, 997, 0, 1);
        RealMatrix testX = x.getSubMatrix(998, 1249, 0, 1);
        ArrayRealVector y = new ArrayRealVector(smarket.getDirection());
        RealVector trainY = y.getSubVector(0, 998);
        RealVector testY = y.getSubVector(998, 252);

        QDA model = new QDA(trainX.getData(), trainY.toArray());
        ArrayRealVector yHat = new ArrayRealVector(Macro.toArray(Arrays.stream(testX.getData()).map(model::predict).collect(Collectors.toList())));
        ConfusionMatrix confusion = new ConfusionMatrix(yHat.getDataRef(), testY.toArray());
        System.out.println(confusion);
    }

    /**
     * p164
     */
    @Test
    public void knnForsmarket() {
        Array2DRowRealMatrix x = Macro.matrixHConcat(smarket.getLag1(), smarket.getLag2());
        RealMatrix trainX = x.getSubMatrix(0, 997, 0, 1);
        RealMatrix testX = x.getSubMatrix(998, 1249, 0, 1);
        ArrayRealVector y = new ArrayRealVector(smarket.getDirection());
        RealVector trainY = y.getSubVector(0, 998);
        RealVector testY = y.getSubVector(998, 252);

        int[] k = new int[]{1, 3};
        for (int i : k) {
            KNN model = new KNN(trainX.getData(), trainY.toArray(), i);
            ArrayRealVector yHat = new ArrayRealVector(model.predict(testX.getData()));
            ConfusionMatrix confusion = new ConfusionMatrix(yHat.getDataRef(), testY.toArray());
            System.out.println(confusion);
        }

    }

    /**
     * p165, k=1 close enough, but k =3/5 not good as textbook
     */
    @Test
    public void knnForCaravan() {
        double[][] raw = caravan.getX();
        double[] y = caravan.getY();
        double[][] x = new double[5822][85];
        x = Arrays.stream(raw).map(StatUtils::normalize).collect(Collectors.toList()).toArray(x);
        double[][] testX = new Array2DRowRealMatrix(x).getSubMatrix(0, 999, 0, 84).getData();
        double[] testY = new ArrayRealVector(y).getSubVector(0, 1000).toArray();
        double[][] trainX = new Array2DRowRealMatrix(x).getSubMatrix(1000, 5821, 0, 84).getData();
        double[] trainY = new ArrayRealVector(y).getSubVector(1000, 4822).toArray();

        KNN model = new KNN(trainX, trainY, 3);
        double[] yHat = model.predict(testX);

        ConfusionMatrix confusion = new ConfusionMatrix(yHat, testY);
        System.out.println(confusion);
        double specialErrorRate = confusion.getTruePositive() / ((double)confusion.getTruePositive() + confusion.getFalsePositive());
        System.out.println(specialErrorRate);

    }

}
