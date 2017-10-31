


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class DummyDataGenerator {

    public static final String[] listOfGroupedProducts = {"charcoal briquettes", "lighter fluid", "hot dog buns", "hot dogs", "ketchup", "mustard", "pickles", "sausage", "t-bone steak", "veggie tray", "relish", "brown mustard", "spicy mustard", "potato chips", "corn chips", "salsa", "corn", "*", "wood screws", "wood glue", "wd40", "white bathroom caulk", "beige caulk", "small hammer", "nails", "sandpaper", "door stop", "*", "potato chips", "corn chips", "chocolate covered pretzels", "popcorn", "hummus", "guacamole", "brownies", "salsa", "*", "shovel", "fertilizer", "potting soil", "mulch", "ground cover", "terra cotta pots", "garden hose", "spinach seeds", "wildflower seeds", "pansies", "*", "princess leia doll", "han solo action figure", "star wars napkins", "star wars balloons", "luke skywalker pillow", "darth vader action figure", "rogue-1 toys", "*", "toilet paper", "paper towels", "dawn dish soap", "shrek napkins", "7th generation dish soap", "tide laundry detergent", "all laundry detergent", "*", "birthday candles", "shrek napkins", "streamers", "shrek balloons", "shrek paper plates", "cake", "*", "birthday candles", "star wars napkins", "star wars plates", "cake", "han solo action figure", "darth vader action figure"};

    public static final Instant initialInstant = Instant.now().minus(Duration.ofDays(7));

    public static Map<String, String> m_prodId_prodTitle;

    public static final int orderPercentage = 30;
//// Adding 5 hours and 4 minutes to an Instant
//instant.plus(Duration.ofHours(5).plusMinutes(4));

    /**
     * what is purpose right now ...
     * <p>
     * there be two data sources.
     * <p>
     * 1.  elasticsearch - product id, product title, prod description, prod image url.
     * 2.  browseOrderData - set/list of rows:  customerId, eventType (browse or order) (use boolean), productId, timestamp
     * no - this should be map (key: customer, value: CustomerBrowseOrderData)
     * <p>
     * <p>
     * ^ generate artificial data for both these.
     * <p>
     * how will browseOrderData exist?  starts as parquet files.  but i want it in memory.  want to use to find boughtTogether
     * <p>
     * boughtTogether - input: set of products.  output: list of sets of products in decreasing relevance.  relevance: number of instances bought together.
     * <p>
     * just model boughtTogether data as set of BrowseOrOrderEvent objects ( customerId, eventType (browse or order), productId, timestamp).
     * iterate through Event objects per product to determine boughtTogether
     * <p>
     * <p>
     * 1.  for listOfGroupedProducts, put into set and assign productIds, put in map
     * 2.  for listOfGroupedProducts, make list of sets of products, delineated by *
     * <p>
     * make a bunch of customers and generate BrowseOrOrderEvent objects set
     */
    public static void main(String[] asdf) {

        m_prodId_prodTitle = getMap_prodId_prodTitle(listOfGroupedProducts);
        List<List<String>> listOfProductSets = getListOfProductSets(listOfGroupedProducts);


        Set<List<BrowseOrOrderEvent>> browseOrderData = generateCustomerBrowseOrderData(m_prodId_prodTitle, listOfProductSets);

        File f = new File("tempDummyData.csv");

        System.out.println(BrowseOrOrderEvent.getHeader());

        StringBuilder sb = new StringBuilder();

        for (List<BrowseOrOrderEvent> list : browseOrderData) {
            for (BrowseOrOrderEvent event : list) {
                System.out.println(event);
                sb.append(event.toString() + System.lineSeparator());

            }
        }

        try {
            Files.write(f.toPath(), sb.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static Set<List<BrowseOrOrderEvent>> generateCustomerBrowseOrderData(Map<String, String> m_prodId_prodTitle, List<List<String>> listOfProductSets) {
        Set<List<BrowseOrOrderEvent>> browseOrderData = new HashSet<>();

        int numProductSets = listOfProductSets.size();

        int numCustomers = 1000;

        int n = 10;

        boolean value = trueOneOutOfNTimes(n);

        for (int customerId = 1; customerId <= numCustomers; customerId++) {

            List<BrowseOrOrderEvent> customerbrowseOrderEvents = new ArrayList<>();

            int numEvents = ThreadLocalRandom.current().nextInt(1, 20 + 1);

            int customersMainProductSetIndex = ThreadLocalRandom.current().nextInt(0, numProductSets);

            for (int eventNum = 0; eventNum < numEvents; eventNum++) {
                customerbrowseOrderEvents.add(generateNewCustomerBrowseOrOrderEvent(customerId + "", listOfProductSets, customersMainProductSetIndex, m_prodId_prodTitle));
            }
            Collections.sort(customerbrowseOrderEvents);
            browseOrderData.add(customerbrowseOrderEvents);
        }


        return browseOrderData;
    }

    private static BrowseOrOrderEvent generateNewCustomerBrowseOrOrderEvent(String customerId, List<List<String>> listOfProductSets, int customersMainProductSetIndex, Map<String, String> m_prodId_prodTitle) {


        int productSetIndex = getProductSetIndex_P_outOf100_probabilityIndexI(80, customersMainProductSetIndex, listOfProductSets.size());

        List<String> productSet = listOfProductSets.get(productSetIndex);

//        System.out.println("productSet:");
//        System.out.println(productSet);

        String productTitle = getRandomProductFromList(productSet);

        boolean isOrder = percent(orderPercentage);

        Instant eventInstant = initialInstant.plus(Duration.ofMinutes(ThreadLocalRandom.current().nextInt(1, 60 * 24 * 7)));

        String productId = getKey(m_prodId_prodTitle, productTitle);


        return new BrowseOrOrderEvent(customerId, isOrder, productId, eventInstant);
    }

    private static String getKey(Map<String, String> map, String value) {


        for (String key : map.keySet()) {
            if (map.get(key).equals(value)) {
                return key;
            }
        }
        System.out.println("map: " + map);
        System.out.println("value: <" + value + ">");
        throw new RuntimeException("key never found. " + value);
    }

    private static int getProductSetIndex_P_outOf100_probabilityIndexI(int p, int i, int numProductSets) {


        if (percent(p)) {
            return i;
        } else {

            int answer;
            do {
                answer = ThreadLocalRandom.current().nextInt(0, numProductSets);
            } while (answer == i && numProductSets > 1);
            return answer;

        }
    }

    public static boolean percent(int percentage) {
        return ThreadLocalRandom.current().nextInt(1, 100) < percentage;
    }

    private static String getRandomProductFromList(List<String> productList) {

        int productSetSize = productList.size();

        int productIndex = ThreadLocalRandom.current().nextInt(0, productSetSize);

        String productTitle = productList.get(productIndex);

        return productTitle;
    }

    private static boolean trueOneOutOfNTimes(int n) {
        int randomNum = ThreadLocalRandom.current().nextInt(1, n + 1);
        return randomNum == 1;
    }


    public static Map<String, String> getMap_prodId_prodTitle() {
        return getMap_prodId_prodTitle(listOfGroupedProducts);

    }

    private static Map<String, String> getMap_prodId_prodTitle(String[] listOfGroupedProducts) {

        TreeSet<String> titles = new TreeSet<>();   //for fixed order

        for (String title : listOfGroupedProducts) {
            titles.add(title);
        }
        titles.remove("*");

        //make map with key = productId
        Map<String, String> m_prodId_prodTitle = new HashMap<>();
        int key = 0;
        for (String title : titles) {
            m_prodId_prodTitle.put(key++ + "", title);
        }
        return m_prodId_prodTitle;
    }

    private static List<List<String>> getListOfProductSets(String[] listOfGroupedProducts) {


        List<List<String>> listOfProductGroupings = new ArrayList<>();

        String listJoined = String.join(",", listOfGroupedProducts);

        String[] groups = listJoined.split("\\*");

        for (String group : groups) {
            List<String> set = new ArrayList<>();
            String[] _titles = group.split(",");
            for (String title : _titles) {
                if (!title.isEmpty()) {
                    set.add(title);
                }
            }
            listOfProductGroupings.add(set);
        }
        return listOfProductGroupings;
    }

    static class BrowseOrOrderEvent implements Comparable<BrowseOrOrderEvent> {
        public String customerId;
        public boolean eventType;   //true for order
        public String productId;
        public Instant timestamp;

        public BrowseOrOrderEvent(String customerId, boolean eventType, String productId, Instant timestamp) {
            this.customerId = customerId;
            this.eventType = eventType;
            this.productId = productId;
            this.timestamp = timestamp;
        }

        public String toString() {
            return customerId + "," + eventType + "," + productId + "," + m_prodId_prodTitle.get(productId) + "," + timestamp.toString();
        }

        @Override
        public int compareTo(BrowseOrOrderEvent other) {
            return this.timestamp.compareTo(other.timestamp);
        }

        public static String getHeader() {
            return "customerId, isOrder?, productId, productTitle, timestamp";
        }
    }

    static class CustomerBrowseOrOrderEvent implements Comparable<CustomerBrowseOrOrderEvent> {
        public boolean eventType;   //true for order
        public String productId;
        public Instant timestamp;

        public CustomerBrowseOrOrderEvent(boolean eventType, String productId, Instant timestamp) {
            this.eventType = eventType;
            this.productId = productId;
            this.timestamp = timestamp;
        }

        @Override
        public int compareTo(CustomerBrowseOrOrderEvent other) {
            return this.timestamp.compareTo(other.timestamp);
        }
    }
}
