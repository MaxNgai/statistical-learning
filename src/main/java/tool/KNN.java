package tool;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Max Ngai
 * @since 2022/11/1
 */
public class KNN {
    private double[][] X;
    private double[] Y;
    private double[] yHat;
    private int p;
    private int n;
    private int k;

    /**
     *
     * @param x
     * @param y
     * @param k k points nearest selected
     */
    public KNN(double[][] x, double[] y, int k) {
        this.X = x;
        this.Y = y;
        this.n = x.length;
        this.p = x[0].length;
        this.k = k;
    }

    public double predict(double[] input) {
        ArrayRealVector x0 = new ArrayRealVector(input);

        List<Pair<Double, Double>> distance = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            ArrayRealVector a = new ArrayRealVector(X[i]);
            ArrayRealVector dev = a.subtract(x0);
            double d = dev.dotProduct(dev);
            distance.add(Pair.create(d, Y[i]));
        }
        distance.sort(Comparator.comparing(Pair::getFirst));
        List<Pair<Double, Double>> sublist = distance.subList(0, k);
        Map<Double, List<Pair<Double, Double>>> map = sublist.stream().collect(Collectors.groupingBy(Pair::getSecond));

        double res = Double.MIN_VALUE;
        double obs = -1;

        for (Map.Entry<Double, List<Pair<Double, Double>>> entry : map.entrySet()) {
            if (entry.getValue().size() > obs) {
                obs = entry.getValue().size();
                res = entry.getKey();
            }
        }

        return res;

    }

}
