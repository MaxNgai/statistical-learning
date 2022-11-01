package exercise;

import data.Default;
import data.Smarket;
import org.apache.commons.math3.analysis.function.Logistic;
import org.apache.commons.math3.analysis.function.Logit;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.distribution.LogisticDistribution;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.junit.Test;
import tool.Covariance;
import tool.LDA;
import tool.Macro;
import tool.QDA;

import java.util.Arrays;

/**
 * @author Max Ngai
 * @since 2022/10/17
 */
public class Classification {

    Default aDefault = new Default();

    Smarket smarket = new Smarket();

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


}
