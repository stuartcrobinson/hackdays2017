package example;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * @author Jakub Podlesak (jakub.podlesak at oracle.com)
 */
@XmlRootElement(name = "change")
public class ChangeRecordBean {

    @XmlAttribute
    public boolean madeByAdmin;
    public int linesChanged;
    public String logMessage;

    /**
     * No-arg constructor for JAXB
     */
    public ChangeRecordBean() {}

    public ChangeRecordBean(boolean madeByAdmin, int linesChanged, String logMessage) {
        this.madeByAdmin = madeByAdmin;
        this.linesChanged = linesChanged;
        this.logMessage = logMessage;
    }
}
