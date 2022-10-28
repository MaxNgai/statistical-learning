package data;

import org.apache.commons.lang3.StringUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
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
