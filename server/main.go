package main

import "net/http"
import "log"
import "os"
import "math/rand"
import "io"
import "strconv"

const PRIVATE_ROOT = "/var/www"

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case PUBLIC_INDEX_PATH:
			switch r.Method {
			case "GET":
				serveIndexGet(w, r)
			case "POST":
				serveIndexPost(w, r)
			}
		default:
			serveNotFound(w, r)
		}
	})
	log.Fatal(http.ListenAndServe(":80", nil))
}

const PUBLIC_INDEX_PATH = "/"
const PRIVATE_INDEX_PATH = PRIVATE_ROOT + "/index.html"

func serveIndexGet(w http.ResponseWriter, r *http.Request) {
	file, err := os.Open(PRIVATE_INDEX_PATH)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	fileInfo, err := file.Stat()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	modTime := fileInfo.ModTime()
	http.ServeContent(w, r, PRIVATE_INDEX_PATH, modTime, file)
}

const BYTES_IN_GIGABYTE = 2^30
const UPLOAD_MEMORY_BYTES_MAX = 0.25 * BYTES_IN_GIGABYTE
const UPLOAD_ROOT = "/var/uploads"
const IMAGES_FIELD_NAME = "images"
const RW_RW_R = 0664

func serveIndexPost(w http.ResponseWriter, r *http.Request) {
	err := r.ParseMultipartForm(UPLOAD_MEMORY_BYTES_MAX)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	m := r.MultipartForm
	files := m.File[IMAGES_FIELD_NAME]
	thisUploadRoot := UPLOAD_ROOT + "/" + strconv.Itoa(rand.Int()) + "/"
	os.Mkdir(thisUploadRoot, os.ModeDir)
	os.Chmod(thisUploadRoot, RW_RW_R)
	thisUploadImagesRoot := thisUploadRoot + "images/"
	os.Mkdir(thisUploadImagesRoot, os.ModeDir)
	os.Chmod(thisUploadImagesRoot, RW_RW_R)
	for i, _ := range files {
		file, err := files[i].Open()
		defer file.Close()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		dst, err := os.Create(thisUploadImagesRoot + files[i].Filename)
		os.Chmod(thisUploadImagesRoot + files[i].Filename, RW_RW_R)
		defer dst.Close()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if _, err := io.Copy(dst, file); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	emailFile, err := os.Create(thisUploadRoot + "email.txt")
	defer emailFile.Close()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	emailFile.WriteString(r.PostFormValue("email"))
}

func serveNotFound(w http.ResponseWriter, r *http.Request) {
	http.NotFound(w, r)
}