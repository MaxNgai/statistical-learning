package tool;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.util.*;
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
}
