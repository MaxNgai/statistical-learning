package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * @author Max Ngai
 * @since 2022/10/11
 */
@Data
public class Credit {
    private Array2DRowRealMatrix data;

    public Credit() {
        Array2DRowRealMatrix credit = DataReader.read("Credit");
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

}
