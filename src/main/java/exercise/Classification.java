package exercise;

import data.Default;
import org.apache.commons.math3.analysis.function.Logistic;
import org.apache.commons.math3.analysis.function.Logit;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.distribution.LogisticDistribution;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.junit.Test;
import tool.LDA;
import tool.Macro;

import java.util.Arrays;

/**
 * @author Max Ngai
 * @since 2022/10/17
 */
public class Classification {

    Default aDefault = new Default();

    @Test
    public void lda() {
        LDA lda = new LDA(Macro.matrixHConcat(aDefault.getStudent(), aDefault.getBalance()).getData(),
                aDefault.getDefault());

        System.out.println(lda.getConfusionMatrix());
        System.out.println(Arrays.stream(lda.getYHat()).sum());
    }

}
