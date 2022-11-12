package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * @author Max Ngai
 * @since 2022/10/13
 */
@Data
public class Boston {
    private Array2DRowRealMatrix data;

    public Boston() {
        data = DataReader.read("Boston", new DefaultParser());
    }

    public double[] getCrime() {
        return data.getColumn(1);
    }
    public double[] getMedv() {
        return data.getColumn(13);
    }

    public double[][] getX() {
        return data.getSubMatrix(0, data.getRowDimension() - 1, 2, data.getColumnDimension() - 1).getData();
    }
}
