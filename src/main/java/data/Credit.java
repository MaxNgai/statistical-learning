package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Objects;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/10/11
 */
@Data
public class Credit {
    private Array2DRowRealMatrix data;

    public Credit() {
        Array2DRowRealMatrix credit = DataReader.read("Credit", new CreditParser());
        data = credit;
    }

    public double[] getBalance() {
        return data.getColumn(10);
    }

    public double[] getLimit() {
        return data. getColumn(1);
    }

    public double[] getRating() {
        return data.getColumn(2);
    }

    public double[] getAge() {
        return data.getColumn(4);
    }

    public RealMatrix getX() {
        RealMatrix subMatrix = data.getSubMatrix(IntStream.range(0, data.getRowDimension()).toArray(),
                IntStream.range(0, data.getColumnDimension()).filter(i -> i < 9).toArray());
        return subMatrix;
    }

    public RealVector getY() {
        return new ArrayRealVector(getBalance());
    }

    private static class CreditParser extends DefaultParser {
        @Override
        public double parse(String raw, int columnNo) {
            if (columnNo == 9) {
                return -1;
            }

            if (Objects.equals(raw, "Yes")) {
                return 1D;
            } else if(Objects.equals(raw, "No")) {
                return 0D;
            } else {
                return super.parse(raw, columnNo);
            }
        }
    }
}
