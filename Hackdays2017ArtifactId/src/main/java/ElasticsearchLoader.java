import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ElasticsearchLoader {

  public static class es {
    public static String url = "http://robinson.brontolabs.local:9115";
  }

  public static class f {
    public static String product = "product";
    public static String products = "products";
    public static String productId = "productId";
    public static String imageUrl = "imageUrl";
    public static String description = "description";
    public static String title = "title";
    public static String titlesuggest = "titlesuggest";
    public static String vv_indicators = "vv_indicators";
  }

  public static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");
  static OkHttpClient client = new OkHttpClient();

  public static final JSONObject getMapping() {
    return new JSONObject()
        .put("mappings", new JSONObject()
            .put(f.product, new JSONObject()
                .put("properties", new JSONObject()
                    .put(f.productId, new JSONObject().put("type", "keyword"))
                    .put(f.imageUrl, new JSONObject().put("type", "text").put("index", "false"))
                    .put(f.description, new JSONObject().put("type", "text"))
                    .put(f.title, new JSONObject().put("type", "text"))
                    .put(f.titlesuggest, new JSONObject().put("type", "completion"))
                    .put(f.vv_indicators, new JSONObject().put("type", "keyword")))));
  }

  public static JSONObject createProductsIndex() {
    return put(es.url + "/" + f.products, getMapping());
  }

  public static JSONObject deleteProductsIndex() {
    return delete(es.url + "/" + f.products);
  }

  public static void main(String[] args) {
    rebuildProductsIndex();
  }

  public static void rebuildProductsIndex() {

    try {
      System.out.println(deleteProductsIndex());
      System.out.println(createProductsIndex());
//            System.exit(0);

      Map<String, String> m_prodId_prodTitle = DummyDataGenerator.getMap_prodId_prodTitle();

//            Map<String, String> m_prodId_prodTitle = new HashMap<>();
//            int count = 0;
//            for (String key : m_prodId_prodTitle0.keySet()) {
//                m_prodId_prodTitle.put(key, m_prodId_prodTitle0.get(key));
//                if (count++ > 2) break;
//            }

      Map<String, String> m_prodId_imageUrl = BingImageUrlFetcher.getBingImageUrl(m_prodId_prodTitle);

      List<String> fileLines = new ArrayList<>();

      int count = 0;

      Random rand = new Random();

      List<String> productIds = new ArrayList(m_prodId_prodTitle.keySet());

      for (String productId : m_prodId_prodTitle.keySet()) {

        String productTitle = m_prodId_prodTitle.get(productId);
        String productDescription = "The best " + productTitle + " money can buy!";
        String imageUrl = m_prodId_imageUrl.get(productId);

        List<String> list = new ArrayList<>();
        list.add(productIds.get(rand.nextInt(productIds.size() - 1) + 0));
        list.add(productIds.get(rand.nextInt(productIds.size() - 1) + 0));
        list.add(productIds.get(rand.nextInt(productIds.size() - 1) + 0));
        list.add(productIds.get(rand.nextInt(productIds.size() - 1) + 0));
        list.add(productIds.get(rand.nextInt(productIds.size() - 1) + 0));

        JSONObject json = new JSONObject()
            .put(f.productId, productId)
            .put(f.title, productTitle)
            .put(f.titlesuggest, productTitle)
            .put(f.description, productDescription)
            .put(f.vv_indicators, new JSONArray(list))
            .put(f.imageUrl, imageUrl);

        fileLines.add(productId + "," + productTitle + "," + productDescription + "," + imageUrl);

        JSONObject result = post(es.url + "/" + f.products + "/" + f.product, json);
        System.out.println(result);
      }

      Files.write(new File("dummyData.csv").toPath(), fileLines, Charset.defaultCharset());


    } catch (IOException e) {
      e.printStackTrace();
    }
//
//        PUT products/product/_update
//        {
//        "productId" : "PROD 001",
//        "imageUrl" : "http://www.asdf.com/1.jpg",
//        "description" : "the best toaster",
//        "title" : "toaster"
//        }


    /*

    https://yesno.wtf/api
     */
//
//        try {
//            String asdf = Unirest.get("http://httpbin.org/{method}")
//                    .routeParam("method", "get")
//                    .queryString("name", "Mark")
//                    .asJson()
//                    .toString();
//            System.out.println(asdf);
//        } catch (UnirestException e) {
//            e.printStackTrace();
//        }

  }


  public static JSONObject post(String url, Object json) {
    try {
      RequestBody body = RequestBody.create(JSON, json.toString());
      Request request = new Request.Builder()
          .url(url)
          .post(body)
          .build();
      Response response = client.newCall(request).execute();
      return new JSONObject(response.body().string());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static JSONObject put(String url, Object json) {
    try {
      RequestBody body = RequestBody.create(JSON, json.toString());
      Request request = new Request.Builder()
          .url(url)
          .put(body)
          .build();

      Response response = client.newCall(request).execute();
      return new JSONObject(response.body().string());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static JSONObject delete(String url) {
    try {
      Request request = new Request.Builder()
          .url(url)
          .delete()
          .build();
      Response response = client.newCall(request).execute();
      return new JSONObject(response.body().string());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static JSONObject get(String url) {
    try {
      Request request = new Request.Builder()
          .url(url)
          .build();

      Response response = null;
      response = client.newCall(request).execute();
      return new JSONObject(response.body().string());

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

}

//
//    DELETE products
//
//    PUT products
//{
//        "mappings": {
//        "product" : {
//        "properties" : {
//        "productId" : {
//        "type" : "keyword",
//        "enabled": false
//        },
//        "imageUrl" : {
//        "type" : "text",
//        "index" : "false"
//        },
//        "description" : {
//        "type" : "text"
//        },
//        "title" : {
//        "type": "text"
//        }
//        }
//        }
//        }
//        }
//
//
//        PUT products/product/_update
//        {
//        "productId" : "PROD 001",
//        "imageUrl" : "http://www.asdf.com/1.jpg",
//        "description" : "the best toaster",
//        "title" : "toaster"
//        }
