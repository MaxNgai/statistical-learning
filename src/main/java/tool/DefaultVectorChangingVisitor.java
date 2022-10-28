package tool;

import lombok.Data;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;

import java.util.function.Function;

/**
 * tool for {@link org.apache.commons.math3.linear.RealMatrix#walkInColumnOrder}
 *
 * do a transformation for every element in the matrix
 *
 * @author Max Ngai
 * @since 2022/10/13
 */
@Data
public class DefaultVectorChangingVisitor implements RealVectorChangingVisitor {
    protected int dim;
    protected int start;
    protected int end; // inclusive

    protected Function<Double, Double> mapper;

    public DefaultVectorChangingVisitor(Function<Double, Double> mapper) {
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
        return mapper.apply(value);
    };

    @Override
    public double end() {
        return 0;
    }
}
