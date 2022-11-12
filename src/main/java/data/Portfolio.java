package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * @author Max Ngai
 * @since 2022/11/12
 */
@Data
public class Portfolio {
    private Array2DRowRealMatrix data;

    public Portfolio() {
        data = DataReader.read("Portfolio");
    }

    public double[] getX() {
        return data.getColumn(0);
    }

    public double[] getY() {
        return data.getColumn(1);
    }

}
