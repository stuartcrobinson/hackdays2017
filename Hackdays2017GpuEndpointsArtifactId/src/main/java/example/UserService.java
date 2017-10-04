package example;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.core.Response;

@Path("/User")
public class UserService {

  // Single Parameter
  @GET
  @Path("/1/{name}")
  public Response getInfoUser(@PathParam("name") String name) {
//    return Response.status(200).entity("User [name: " + name + "]").build();
//    return Response.status(200).entity("User [name: " + name + "]").build();
    return Response.status(200).entity("{\"urls\": [\"one\", \"two\"]}").build();
  }

}
