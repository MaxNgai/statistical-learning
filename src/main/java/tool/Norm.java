package tool;

import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;

import java.util.Random;

/**
 * like norm() in R
 * it is for generating random number
 *
 * @author Max Ngai
 * @since 2022/10/12
 */
public class Norm {
    private static GaussianRandomGenerator g = new GaussianRandomGenerator(new JDKRandomGenerator(12));

    public static double[] rnorm(int i) {
        double[] res = new double[i];
        for (int j = 0; j < i; j++) {
            res[j] = g.nextNormalizedDouble();
        }
        return res;
    }

    public static double[] runIf(int i) {
        Random r = new Random();
        double[] res = new double[i];
        for (int j = 0; j < i; j++) {
            res[j] = r.nextDouble();
        }
        return res;
    }
}
