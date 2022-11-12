package tool;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/12
 */
public class CrossValidation {

    public static Pair<RealMatrix, RealMatrix> validationSet(RealMatrix m) {
        List<double[]> train = new ArrayList<>();
        List<double[]> test = new ArrayList<>();

        List<Integer> collect = IntStream.range(0, m.getRowDimension()).boxed().collect(Collectors.toList());
//        Collections.shuffle(collect);

        for (int i = 0; i < collect.size(); i++) {
            if (i < m.getRowDimension() / 2) {
                train.add(m.getRow(collect.get(i)));
            } else {
                test.add(m.getRow(collect.get(i)));
            }
        }

        return Pair.create(
                Macro.vstack(train),
                Macro.vstack(test)
        );

    }

    /**
     * use loocv to yield the mse
     * @return
     */
    public static double loocvMse(RealMatrix x, RealVector y, CvMseGetter getMse) {
        int n = y.getDimension();
        int p = x.getColumnDimension();
        List<Integer> rows = IntStream.range(0, n).boxed().collect(Collectors.toList());
        int[] selectedColumns = IntStream.range(0, p).toArray();

        return IntStream.range(0, n)
                .boxed()
                .map(i -> {
                    ArrayList<Integer> removed = new ArrayList<>(rows);
                    removed.remove(i);
                    int[] selectedRows = removed.stream().mapToInt(e -> e).toArray();
                    RealMatrix trainX = x.getSubMatrix(selectedRows, selectedColumns);
                    RealMatrix testX = x.getSubMatrix(new int[]{i}, selectedColumns);
                    ArrayRealVector testY = new ArrayRealVector(new double[]{y.getEntry(i)});
                    List<Double> rawTrainY = Arrays.stream(y.toArray()).boxed().collect(Collectors.toList());
                    rawTrainY.remove(i.intValue());
                    ArrayRealVector trainY = new ArrayRealVector(Macro.toArray(rawTrainY));

                    double mse = getMse.testSetMse(trainX, trainY, testX, testY);
                    return mse;
                }).mapToDouble(e -> e).average().getAsDouble();

    }

    /**
     * k-fold to yield mse
     * @param x
     * @param y
     * @param getMse
     * @param k
     * @return
     */
    public static double kFoldCv(RealMatrix x, RealVector y, CvMseGetter getMse, int k) {
        int n = y.getDimension();
        int p = x.getColumnDimension();
        int subsetSize = n / k;
        List<Integer> rows = IntStream.range(0, n).boxed().collect(Collectors.toList());
        int[] selectedColumns = IntStream.range(0, p).toArray();
        int kPai = k * subsetSize < n ? k + 1 : k; // in case n % k != 0

        return IntStream.range(0, kPai)
                .boxed()
                .map(i -> {
                    try {

                    List<Integer> testIndex = rows.subList(i * subsetSize, Math.min((i + 1) * subsetSize, n));
                    ArrayList<Integer> trainIndex = new ArrayList<>(rows);
                    trainIndex.removeAll(testIndex);
                    int[] selectedRows = trainIndex.stream().mapToInt(e -> e).toArray();
                    int[] selectedTestRows = testIndex.stream().mapToInt(e -> e).toArray();
                    RealMatrix trainX = x.getSubMatrix(selectedRows, selectedColumns);
                    RealMatrix testX = x.getSubMatrix(selectedTestRows, selectedColumns);

                    RealVector testY = y.getSubVector(i * subsetSize, Math.min(subsetSize, n - i * subsetSize));
                    RealVector trainY;
                    if (i == 0) {
                        trainY = y.getSubVector((i + 1) * subsetSize, n - subsetSize);
                    } else if (i == kPai - 1) {
                        trainY = y.getSubVector(0, i * subsetSize);
                    } else {
                        RealVector left = y.getSubVector(0, i * subsetSize);
                        RealVector right = y.getSubVector((i + 1) * subsetSize, n - ((i + 1) * subsetSize));
                        trainY = left.append(right);
                    }

                    double mse = getMse.testSetMse(trainX, trainY, testX, testY);
                    return mse;
                    } catch (Exception e) {
                        System.out.println(i);
                        throw e;
                    }

                }).mapToDouble(e -> e).average().getAsDouble();
    }

    /**
     * functional-interface that use trainX & train Y to train model,
     * the see the mse on testSet(testX, testY)
     */
    public interface CvMseGetter {

        <TEX extends RealMatrix,
        TEY extends RealVector,
        TRX extends RealMatrix,
        TRY extends RealVector>
        double testSetMse(TRX trainX, TRY trainY, TEX testX, TEY testY);
    }
}
