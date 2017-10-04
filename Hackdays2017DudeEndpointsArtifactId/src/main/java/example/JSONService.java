package example;

import org.json.simple.JSONArray;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//import java.util.function.Predicate;

@Path("/json/metallica")
public class JSONService {

  static int count = 0;

  @GET
  @Path("/get/{name}")
  @Produces(MediaType.APPLICATION_JSON)
  public Response getOptions(@PathParam("name") String name) {

    System.out.println("query recieved!!! "+((count++) % 10)+"-> " + name);

    name = name.toLowerCase().replaceAll(" ", "");

    final String name2 = name;

    List<String> list = (List) Arrays.asList(gifLinksAr);
    List<String> resultsList = new ArrayList<String>();

    for (String s : gifLinksAr) {
      if (s.toLowerCase().contains(name))
        resultsList.add(s);
    }

    JSONArray mJSONArray = new JSONArray();
    mJSONArray.addAll(resultsList);

    String asdf = mJSONArray.toJSONString();

    return Response.ok()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "POST, GET, PUT, UPDATE, OPTIONS")
        .header("Access-Control-Allow-Headers", "Content-Type, Accept, X-Requested-With")
        // .entity("{\"urls\": [\"one\", \"two\"]}")
        .entity(asdf)
        // .entity("lalalala, lalalal! oh yeah cool 'nice' D*&FW(#IUFW name: " + name + "\n" + asdf)
        .build();
  }

  @GET
  @Path("/get/")
  @Produces(MediaType.APPLICATION_JSON)
  public Response getOptions() {

    System.out.println("query recieved!!! "+((count++) % 10)+"-> ");

    List<String> list = (List) Arrays.asList(gifLinksAr);

    JSONArray mJSONArray = new JSONArray();
    mJSONArray.addAll(list);

    String asdf = mJSONArray.toJSONString();

    return Response.ok()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "POST, GET, PUT, UPDATE, OPTIONS")
        .header("Access-Control-Allow-Headers", "Content-Type, Accept, X-Requested-With")
        // .entity("{\"urls\": [\"one\", \"two\"]}")
        .entity(asdf)
        // .entity("lalalala, lalalal! oh yeah cool 'nice' D*&FW(#IUFW name: " + name + "\n" + asdf)
        .build();
  }

  String[] gifLinksAr = new String[] {
      "http://bestanimations.com/Animals/Mammals/Dogs/puppies/adorable-cute-funny-dog-puppy-animated-gif-17.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-56.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-46.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-38.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-32.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-29.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-15.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-14.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-51.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-54.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-33.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-58.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-25.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-5.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-4.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-3.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-35.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-24.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-47.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-48.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-52.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-13.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-53.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-49.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-39.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-12.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-17.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-7.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-55.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-30.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-20.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-2.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-57.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-18.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-22.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-23.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-11.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-43.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-19.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-6.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-44.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-9.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-34.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-8.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-41.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-37.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-50.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-45.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-42.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-28.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-27.gif",
      "http://bestanimations.com/Animals/Mammals/Dogs/dogs/cute-funny-dog-animated-gif-16.gif"

  };
}
