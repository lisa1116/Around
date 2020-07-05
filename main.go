package main

import (
	"cloud.google.com/go/storage"
	"context"
	"encoding/json"
	"fmt"
	jwtmiddleware "github.com/auth0/go-jwt-middleware"
	jwt "github.com/dgrijalva/jwt-go"
	"github.com/gorilla/mux"
	"github.com/olivere/elastic"
	"github.com/pborman/uuid"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"reflect"
	"strconv"
)

const (
	POST_INDEX = "post"
	DISTANCE   = "200km"

	ES_URL = "http://10.128.0.2:9200"

	BUCKET_NAME = "lusha-bucket"
	API_PREFIX  = "/api/v1"
)

// support different types of media files, this is a hash map
var (
	mediaTypes = map[string]string{
		".jpeg": "image",
		".jpg":  "image",
		".gif":  "image",
		".png":  "image",
		".mov":  "video",
		".mp4":  "video",
		".avi":  "video",
		".flv":  "video",
		".wmv":  "video",
	}
)

type Location struct {
	Lat float64 `json:"lat"`
	Lon float64 `json:"lon"`
}

type Post struct {
	User     string   `json:"user"`
	Message  string   `json:"message"`
	Location Location `json:"location"`
	Url      string   `json:"url"`
	Type     string   `json:"type"`
	Face     float32  `json:"face"`
}

func main() {
	fmt.Println("started-service")
	jwtMiddleware := jwtmiddleware.New(jwtmiddleware.Options{
		ValidationKeyGetter: func(token *jwt.Token) (interface{}, error) {
			return []byte(mySigningKey), nil
		},
		SigningMethod: jwt.SigningMethodHS256,
	})

	r := mux.NewRouter()

	r.Handle(API_PREFIX+"/post", jwtMiddleware.Handler(http.HandlerFunc(handlerPost))).Methods("POST", "OPTIONS")
	r.Handle(API_PREFIX+"/search", jwtMiddleware.Handler(http.HandlerFunc(handlerSearch))).Methods("GET", "OPTIONS")
	r.Handle(API_PREFIX+"/cluster", jwtMiddleware.Handler(http.HandlerFunc(handlerCluster))).Methods("GET", "OPTIONS")
	r.Handle(API_PREFIX+"/signup", http.HandlerFunc(handlerSignup)).Methods("POST", "OPTIONS")
	r.Handle(API_PREFIX+"/login", http.HandlerFunc(handlerLogin)).Methods("POST", "OPTIONS")
	log.Fatal(http.ListenAndServe(":8080", r))
}

//upload API
func handlerPost(w http.ResponseWriter, r *http.Request) {
	// Parse from body of request to get a json object.
	fmt.Println("Received one post request")
	w.Header().Set("Content-Type", "application/json")

	//frontend files js, css...are stored in VM, backend allow frontend to access its resource. usee * allows all domain name frontend to access, so interDomain access is allowed. By default is blocked.
	w.Header().Set("Access-Control-Allow-Origin", "*")

	//use authorization header to let backend collect autorization information from frontend request
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type,Authorization")

	//frontend attempt to see if backend can respond interDomain access, a little like handshake
	if r.Method == "OPTIONS" {
		return
	}

	//take token information, and use username as a key to get the real username
	user := r.Context().Value("user")
	claims := user.(*jwt.Token).Claims
	username := claims.(jwt.MapClaims)["username"]

	// read parameters from client
	lat, _ := strconv.ParseFloat(r.FormValue("lat"), 64)
	lon, _ := strconv.ParseFloat(r.FormValue("lon"), 64)

	p := &Post{
		User:    username.(string),
		Message: r.FormValue("message"),
		Location: Location{
			Lat: lat,
			Lon: lon,
		},
	}

	//save image to GCS, FormFile refers to the form-data key type in a post request
	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Image is not available", http.StatusBadRequest)
		fmt.Printf("Image is not available %v\n", err)
		return
	}

	suffix := filepath.Ext(header.Filename)
	if t, ok := mediaTypes[suffix]; ok {
		p.Type = t
	} else {
		p.Type = "unknown"
	}

	// create new unique id for the new object, not readable to human
	id := uuid.New()

	//saveToGCS returns URL
	mediaLink, err := saveToGCS(file, id)
	if err != nil {
		http.Error(w, "Failed to save image to GCS", http.StatusInternalServerError)
		fmt.Printf("Failed to save image to GCS %v\n", err)
		return
	}
	p.Url = mediaLink

	//annotate image with vision api
	//id here is the new id created by uuid()
	//Sprintf() assigns the result to the variable on the left of assignment symbol
	if p.Type == "image" {
		uri := fmt.Sprintf("gs://%s/%s", BUCKET_NAME, id)
		// url -> score is annotate(), implemented by us in vision.go
		if score, err := annotate(uri); err != nil {
			http.Error(w, "Failed to annotate image", http.StatusInternalServerError)
			fmt.Printf("Failed to annotate the image %v\n", err)
			return
		} else {
			p.Face = score
		}
	}

	//save post to ES
	err = saveToES(p, POST_INDEX, id)
	if err != nil {
		http.Error(w, "Failed to save post to Elasticsearch", http.StatusInternalServerError)
		fmt.Printf("Failed to save post to Elasticsearch %v\n", err)
		return
	}

}

//search API
func handlerSearch(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Received one request for search")
	//search also respond with header info, telling fronend what type of data it is responding with
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type,Authorization")

	if r.Method == "OPTIONS" {
		return
	}

	//r.URL.Query represents the search input in URL after the question mark
	//strconv.ParseFloat ()is a type conversion, 64 is double
	//_ represents error, we omit it here since frontend is also from our hand
	lat, _ := strconv.ParseFloat(r.URL.Query().Get("lat"), 64)
	lon, _ := strconv.ParseFloat(r.URL.Query().Get("lon"), 64)

	// range is optional
	// range is in circle
	ran := DISTANCE

	// or frontend can specify other range  val variable makes sure it only exists in if statement scope
	if val := r.URL.Query().Get("range"); val != "" {
		ran = val + "km"
	}
	fmt.Println("range is ", ran)

	//NewGeoDistanceQuery() search based on geolocation
	query := elastic.NewGeoDistanceQuery("location")

	// pass in search properties, Distance() Lat() Lon() are struct components from GeoDistanceQuery()
	query = query.Distance(ran).Lat(lat).Lon(lon)

	// the actual search operation
	searchResult, err := readFromES(query, POST_INDEX)
	if err != nil {
		http.Error(w, "Failed to read post from Elasticsearch", http.StatusInternalServerError)
		fmt.Printf("Failed to read post from Elasticsearch %v.\n", err)
		return
	}

	posts := getPostFromSearchResult(searchResult)

	//Marshall convert golang format to json format
	js, err := json.Marshal(posts)
	if err != nil {
		http.Error(w, "Failed to parse posts into JSON format", http.StatusInternalServerError)
		fmt.Printf("Failed to parse posts into JSON format %v.\n", err)
		return
	}
	// response body requires JSON format
	w.Write(js)
}

func readFromES(query elastic.Query, index string) (*elastic.SearchResult, error) {
	//create new client with ElsticSearch
	client, err := elastic.NewClient(elastic.SetURL(ES_URL))
	if err != nil {
		return nil, err
	}

	//regulate query formats, store returned results in searchResult, as required by ElasticSearch
	searchResult, err := client.Search().
		Index(index).
		Query(query).
		Pretty(true).
		Do(context.Background())
	if err != nil {
		return nil, err
	}
	//why not return &searchResult as specified in signature? becaus Do() returns a pointer to searchResult
	return searchResult, nil
}

func getPostFromSearchResult(searchResult *elastic.SearchResult) []Post {
	var ptype Post
	var posts []Post

	//verify entries that can be casted to ptype
	for _, item := range searchResult.Each(reflect.TypeOf(ptype)) {
		//cast item to Post type
		p := item.(Post)
		posts = append(posts, p)
	}
	return posts
}

// input r is file source
func saveToGCS(r io.Reader, objectName string) (string, error) {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		return "", err
	}

	bucket := client.Bucket(BUCKET_NAME)

	//determine if bucket exists by inspecting bucket attribute
	if _, err := bucket.Attrs(ctx); err != nil {
		return "", err
	}

	object := bucket.Object(objectName)
	wc := object.NewWriter(ctx)
	if _, err := io.Copy(wc, r); err != nil {
		return "", err
	}

	if err := wc.Close(); err != nil {
		return "", err
	}

	// ACL() is access control, to allow all users to read file
	if err := object.ACL().Set(ctx, storage.AllUsers, storage.RoleReader); err != nil {
		return "", err
	}

	attrs, err := object.Attrs(ctx)
	if err != nil {
		return "", err
	}

	fmt.Printf("Image is saved to GCS: %s\n", attrs.MediaLink)

	//mediaLink is the URL returned when file is uploaded successfully
	return attrs.MediaLink, nil
}

// interface can support all data type,
func saveToES(i interface{}, index string, id string) error {
	client, err := elastic.NewClient(elastic.SetURL(ES_URL), elastic.SetSniff(false))
	if err != nil {
		return err
	}

	_, err = client.Index().
		Index(index).
		Id(id).
		BodyJson(i).
		Do(context.Background())

	if err != nil {
		return err
	}

	return nil
}

//search by face score
//Gte: great
func handlerCluster(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Received one cluster request")
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type,Authorization")

	if r.Method == "OPTIONS" {
		return
	}

	term := r.URL.Query().Get("term")
	query := elastic.NewRangeQuery(term).Gte(0.9)

	searchResult, err := readFromES(query, POST_INDEX)
	if err != nil {
		http.Error(w, "Failed to read from Elasticsearch", http.StatusInternalServerError)
		return
	}

	posts := getPostFromSearchResult(searchResult)
	js, err := json.Marshal(posts)
	if err != nil {
		http.Error(w, "Failed to parse post object", http.StatusInternalServerError)
		fmt.Printf("Failed to parse post object %v\n", err)
		return
	}
	w.Write(js)
}
