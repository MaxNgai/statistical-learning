package tool;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * @author Max Ngai
 * @since 2022/10/28
 */
public class Covariance {

    private RealMatrix X;
    private int p;
    private int n;

    private RealMatrix res;


    public Covariance(double[][] input) {
        this(input, true);
    }

    public Covariance(double[][] input, boolean unbias) {
        secondWay(input, unbias);
    }

    /**
     * E[XY] -E[X]E[Y]
     * @param input
     */
    private RealMatrix firstWay(double[][] input, boolean unbias) {
        this.X = new Array2DRowRealMatrix(input);
        n = input.length;
        p = input[0].length;
        int divisor = n;

        res = new Array2DRowRealMatrix(p, p);

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                RealVector v1 = X.getColumnVector(i);
                RealVector v2 = X.getColumnVector(j);
                ArrayRealVector ONE = new ArrayRealVector(n, 1);
                double xyMean = v1.ebeMultiply(v2).dotProduct(ONE) / divisor;
                double xMean = v1.dotProduct(ONE) / divisor;
                double yMean = v2.dotProduct(ONE) / divisor;
                double f = xyMean - xMean * yMean;
                res.setEntry(i, j, unbias ? f * n / (n - 1): f);
            }
        }

        return res;
    }

    /**
     * E[(X-E[X])(Y-E[Y])]
     * @param input
     * @param unbias
     * @return
     */
    public RealMatrix secondWay(double[][] input, boolean unbias) {
        this.X = new Array2DRowRealMatrix(input);
        n = input.length;
        p = input[0].length;
        res = new Array2DRowRealMatrix(p, p);

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                RealVector v1 = X.getColumnVector(i);
                RealVector v2 = X.getColumnVector(j);

                ArrayRealVector ONE = new ArrayRealVector(n, 1);
                double xMean = v1.dotProduct(ONE) / n;
                double yMean = v2.dotProduct(ONE) / n;
                v1.mapSubtractToSelf(xMean);
                v2.mapSubtractToSelf(yMean);
                double f = v1.dotProduct(v2) / (unbias ?  n - 1 : n);
                res.setEntry(i, j, f);
            }
        }

        return res;
    }

    /**
     * covariance matrix for LDA.
     *
     * follow E[(X-E[X])(Y-E[Y)]
     * @param x
     * @param y
     * @return
     */
    public static RealMatrix covForLda(double[][] x, double[] y) {
        Covariance cov = new Covariance();
        cov.X = new Array2DRowRealMatrix(x);
        cov.n = x.length;
        cov.p = x[0].length;
        cov.res = new Array2DRowRealMatrix(cov.p, cov.p);

        // see how many classes
        ArrayList<Double> classes = new ArrayList<>(Arrays.stream(y).boxed().collect(Collectors.toSet()));

        // get mean by classes
        List<RealVector> mean = classes.stream()
                .map(c -> {
                    ArrayRealVector sum = new ArrayRealVector(cov.p, 0D);
                    int count = 0;
                    for (int i = 0; i < y.length; i++) {
                        if (y[i] == c) {
                            sum = sum.add(new ArrayRealVector(x[i]));
                            count++;
                        }
                    }
                    return sum.mapDivide(count);
                }).collect(Collectors.toList());

        for (int i = 0; i < cov.p; i++) {
            for (int j = 0; j < cov.p; j++) {
                RealVector v1 = cov.X.getColumnVector(i);
                RealVector v2 = cov.X.getColumnVector(j);

                AtomicInteger iEntry = new AtomicInteger(i); // java closure doesn't allow refer i directly
                v1.walkInDefaultOrder(new IndexVectorChangingVisitor((index, value) -> {
                    for (int c = 0; c < classes.size(); c++) {
                        if (y[index] == classes.get(c)) {
                            return value - mean.get(c).getEntry(iEntry.get()); // minus different mean according to class
                        }
                    }
                    throw new RuntimeException();
                }));

                AtomicInteger jEntry = new AtomicInteger(j);
                v2.walkInDefaultOrder(new IndexVectorChangingVisitor((index, value) -> {
                    for (int c = 0; c < classes.size(); c++) {
                        if (y[index] == classes.get(c)) {
                            return value - mean.get(c).getEntry(jEntry.get()); // minus different mean according to class
                        }
                    }
                    throw new RuntimeException();
                }));

                double f = v1.dotProduct(v2) / (cov.n); // here is n, rather than n - k, but doesnt affect the final classification result
                cov.res.setEntry(i, j, f);
            }
        }

        return cov.res;
    }

    @Override
    public String toString() {
        return res.toString();
    }

    private Covariance() {}
}
