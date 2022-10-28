package data;

import org.apache.commons.lang3.StringUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * an simple data transform strategy, read anything and convert to double.
 * this class will be used in {@link DataReader#read(String, DefaultParser)}
 * and for mapping data to meet user need.
 *
 * this class will be useful when encounter enumerated input like 'YES' or 'No'.
 * people can use this class to map them to 1 and 0
 *
 * @author Max Ngai
 * @since 2022/10/11
 */
public class DefaultParser {

    /**
     * default data parser
      * @param raw string of an entry
     *  @param columnNo indicates which column
     * @return
     */
    public double parse(String raw, int columnNo) {

        return Double.parseDouble(raw);

    }
}
