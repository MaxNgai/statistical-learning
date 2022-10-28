package tool;

import org.apache.commons.math3.linear.RealVectorChangingVisitor;

import java.util.function.BiFunction;

/**
 * @author Max Ngai
 * @since 2022/10/29
 */
public class IndexVectorChangingVisitor implements RealVectorChangingVisitor {

    protected int dim;
    protected int start;
    protected int end; // inclusive

    protected BiFunction<Integer, Double, Double> mapper;

    public IndexVectorChangingVisitor(BiFunction<Integer, Double, Double> mapper) {
        this.mapper = mapper;
    }

    @Override
    public void start(int dimension, int start, int end) {
        dim = dimension;
        this.start = start;
        this.end = end;
    }

    @Override
    public double visit(int index, double value) {
        return mapper.apply(index, value);
    };

    @Override
    public double end() {
        return 0;
    }
}
