package tool;

import lombok.Data;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;
import tool.model.Model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/12
 */
public class CrossValidation {

    private static ThreadLocalRandom rd = ThreadLocalRandom.current();

    public static Pair<RealMatrix, RealMatrix> validationSet(RealMatrix m) {
        List<double[]> train = new ArrayList<>();
        List<double[]> test = new ArrayList<>();

        List<Integer> collect = IntStream.range(0, m.getRowDimension()).boxed().collect(Collectors.toList());
        Collections.shuffle(collect);

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
    public static double loocvMse(RealMatrix x, RealVector y, Model model) {
        return kFoldCv(x, y, model, y.getDimension());

    }

    /**
     * k-fold to yield mse
     * @param x
     * @param y
     * @param model
     * @param k
     * @return
     */
    public static double kFoldCv(RealMatrix x, RealVector y, Model model, int k) {
        return kFoldDataSplit(x, y, k).stream()
                .map(data -> {
                    return model.train(data.getTrainX(), data.getTrainY()).testMse(data.getTestX(), data.getTestY());
                }).mapToDouble(e -> e).average().getAsDouble();
    }

    /**
     *
     * @param x full set of x
     * @param y full set of y
     * @param k k fold
     * @return list with size of k or k+1!
     */
    public static List<DataSet> kFoldDataSplit(RealMatrix x, RealVector y, int k) {
        int n = y.getDimension();
        int p = x.getColumnDimension();
        int subsetSize = n / k;
        List<Integer> rows = IntStream.range(0, n).boxed().collect(Collectors.toList());
        int[] selectedColumns = IntStream.range(0, p).toArray();
        int kPai = k * subsetSize < n ? k + 1 : k; // in case n % k != 0

        return IntStream.range(0, kPai)
                .boxed()
                .map(i -> {
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

                    return new DataSet(trainX, trainY, testX, testY);
                }).collect(Collectors.toList());

    }

    public static List<Double> bootstrapGetSE(int repeat, RealMatrix input, RealVector output, ParamGetter getter) {
        List<ArrayRealVector> collect = IntStream.range(0, repeat)
                .boxed()
                .map(e -> {
                    RealMatrix x = input.copy();
                    RealMatrix newX = input.copy();
                    RealVector y = output.copy();
                    RealVector newY = output.copy();
                    int n = input.getRowDimension();
                    for (int i = 0; i < n; i++) {
                        // not replacing one row, but all rows
                        int replacement = rd.nextInt(n);
                        newX.setRowVector(i, x.getRowVector(replacement));
                        newY.setEntry(i, y.getEntry(replacement));
                    }
                    List<Double> params = getter.getParams(newX, newY);
                    return new ArrayRealVector(Macro.toArray(params));
                })
                .collect(Collectors.toList());

        RealVector mean = collect.stream().reduce(ArrayRealVector::add)
                .map(e -> e.mapDivide(repeat)).get();

        RealVector res = collect.stream().map(e -> e.subtract(mean))
                .map(e -> e.ebeMultiply(e))
                .reduce(ArrayRealVector::add)
                .map(e -> e.mapDivide(repeat - 1))
                .get();

        res.walkInDefaultOrder(new DefaultVectorChangingVisitor(Math::sqrt));


        return Arrays.stream(res.toArray()).boxed().collect(Collectors.toList());


    }

    /**
     * use X and Y to train model
     * then get estimated params of the model
     */
    public interface ParamGetter {
        <X extends RealMatrix,
         Y extends RealVector>
        List<Double> getParams(X x, Y y);
    }

    @Data
    public static class DataSet {
        private RealMatrix trainX;
        private RealMatrix testX;
        private RealVector trainY;
        private RealVector testY;

        public DataSet(RealMatrix trainX, RealVector trainY, RealMatrix testX, RealVector testY) {
            this.trainX = trainX;
            this.testX = testX;
            this.trainY = trainY;
            this.testY = testY;
        }
    }
}
