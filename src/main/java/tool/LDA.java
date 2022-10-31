package tool;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * home-make lda
 *
 * @author Max Ngai
 * @since 2022/10/27
 */
@Data
public class LDA {

    private static final double YES = 1D;
    private static final double NO = 0D;

    private double[] y;
    private double[][] x;
    private int n;
    private int p;
    private int k;

    private List<Double> classes;
    private RealMatrix covariance;
    private RealMatrix inverseCovariance;
    private double[][] u; // mean vectors
    private double[] pai; // prior probability

    private double[] yHat;

    public LDA(double[][] x, double[] y) {
        this.x = x;
        this.y = y;
        n = y.length;
        p = x[0].length;
        yHat = new double[n];

        classes = new ArrayList<Double>(Arrays.stream(y).boxed().collect(Collectors.toSet()));
        k = classes.size();
        pai = new double[k];
        u = new double[k][p];

        List<ArrayRealVector> sumByClass = classes.stream().map(c -> new ArrayRealVector(p, 0D)).collect(Collectors.toList());
        for (int i = 0; i < k; i++) {
            int obs = 0;
            ArrayRealVector sum = sumByClass.get(i);
            for (int j = 0; j < y.length; j++) {
                if (y[j] == classes.get(i)) {
                    obs++;
                    sum = new ArrayRealVector(this.x[j]).add(sum);
                }
            }
            pai[i] = ((double) obs) / n;
            u[i] = sum.mapDivide(obs).toArray();
        }

        covariance = tool.Covariance.covForLda(x, y); // this cov matrix is not like usual ones, should use different mean by classes
        inverseCovariance= MatrixUtils.inverse(covariance);
        // predict
        for (int i = 0; i < x.length; i++) {
            yHat[i] = predict(x[i]);
        }
    }

    public double predict(double[] x) {
        RealMatrix input = new Array2DRowRealMatrix(x).transpose();

        List<Pair<Integer, Double>> indexAndDiscriminant = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            ArrayRealVector miu = new ArrayRealVector(u[i]);
            double a = input.multiply(inverseCovariance).operate(miu).getEntry(0);
            double b = inverseCovariance.preMultiply(miu).dotProduct(miu) / 2;
            double c = Math.log(pai[i]);

            double discriminant = a - b + c;
            indexAndDiscriminant.add(Pair.create(i, discriminant));
        }

        indexAndDiscriminant.sort(Comparator.comparing(e -> e.getSecond()));

        return classes.get(indexAndDiscriminant.get(classes.size() - 1).getFirst());


    }

    public ConfusionMatrix getConfusionMatrix() {
        int tn = 0;
        int tp = 0;
        int fn = 0;
        int fp = 0;

        for (int i = 0; i < y.length; i++) {
            if (yHat[i] == YES) {
                if (y[i] == YES) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (y[i] == NO) {
                    tn++;
                } else {
                    fn++;
                }
            }
        }

        return ConfusionMatrix.builder()
                .falseNegative(fn)
                .falsePositive(fp)
                .trueNegative(tn)
                .truePositive(tp)
                .build();
    }


}
