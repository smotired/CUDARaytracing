//-------------------------------------------------------------------------------
///
/// \file       xmlload.cpp 
/// \author     Cem Yuksel (www.cemyuksel.com)
/// \version    1.0
/// \date       September 19, 2025
///
/// \brief Example source for CS 6620 - University of Utah.
///
/// Copyright (c) 2019 Cem Yuksel. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or 
/// sublicensing of this code or its derivatives is strictly prohibited.
///
//-------------------------------------------------------------------------------

#include "xmlload.cuh"
#include "../renderer/renderer.cuh"
#include "../scene/objects.cuh"

//-------------------------------------------------------------------------------

Sphere* theSphere; // Will point to managed memory
Plane* thePlane; // Will point to managed memory

//-------------------------------------------------------------------------------

size_t CountNodes( Loader const& loader );
void AssignNodes( Loader const& loader, Node* nodeList, int& next, const Matrix& parentTransform, const Material* materials, unsigned int* materialTable, size_t materialCount );
void LoadLight( Loader const& loader, Light* lightList, int i );
void LoadMaterial( Loader const& loader, Material* materialList, int i, unsigned int* materialTable, size_t materialCount );

std::hash<std::string> hasher;
#define HASH(charptr) hasher(std::string(charptr))

//-------------------------------------------------------------------------------

// Insert into and search a hash table that maps material names to indexes in the materials list.
// The hash table should store 2N ints, where t[2i] is the hash and t[2i+1] is the element.
void InsertMaterial(unsigned int* table, const size_t entryCount, const unsigned int hash, const unsigned int index) {
	// Loop until we find an open spot
	unsigned int i = 2 * (hash % entryCount);
	const unsigned int initI = i;
	while (table[i] != 0) {
		i += 2;
		if (i >= 2 * entryCount) i = 0;
		if (i == initI) throw std::exception(); // no room in table
	}

	// Insert
	table[i] = hash;
	table[i + 1] = index;
}

unsigned int GetMaterial(const unsigned int* table, const size_t entryCount, const unsigned int hash) {
	// Loop until we find the spot
	unsigned int i = 2 * (hash % entryCount);
	const unsigned int initI = i;
	while (table[i] != hash) {
		i += 2;
		if (i >= 2 * entryCount) i = 0;
		if (i == initI) throw std::exception(); // not present in table
	}

	return table[i + 1];
}

//-------------------------------------------------------------------------------

// Similar hash table but for pointers to
#define MAX_MESHES 512
// This is the hash table size, which only lives on the host so it's fine. The actual mesh objects will be all over the place.

void* meshTable[MAX_MESHES * 2];

void InsertMesh(const unsigned long hash, const MeshObject* mesh) {
	// Loop until we find an open spot
	unsigned int i = 2 * (hash % MAX_MESHES);
	const unsigned int initI = i;
	while (meshTable[i] != nullptr) {
		i += 2;
		if (i >= 2 * MAX_MESHES) i = 0;
		if (i == initI) throw std::exception(); // no room in table
	}

	// Insert
	meshTable[i] = (void*)hash;
	meshTable[i + 1] = (void*)mesh;
}

MeshObject* GetMesh(const unsigned long hash) {
	// Loop until we find the spot
	unsigned int i = 2 * (hash % MAX_MESHES);
	const unsigned int initI = i;
	while (meshTable[i] != (void*)hash) {
		i += 2;
		if (i >= 2 * MAX_MESHES) i = 0;
		if (i == initI) return nullptr; // not present in table
	}

	return (MeshObject*)meshTable[i + 1];
}

//-------------------------------------------------------------------------------

bool Renderer::LoadScene( char const *filename )
{
	// Allocate the primitives
	cudaMallocManaged(&theSphere, sizeof(Sphere));
	theSphere = new Sphere();
	cudaMallocManaged(&thePlane, sizeof(Plane));
	thePlane = new Plane();

	// Load the document
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLError e = doc.LoadFile(filename);

	if ( e != tinyxml2::XML_SUCCESS ) {
		printf("ERROR: Failed to load the file \"%s\"\n", filename);
		return false;
	}

	tinyxml2::XMLElement *xml = doc.FirstChildElement("xml");
	if ( ! xml ) {
		printf("ERROR: No \"xml\" tag found.\n");
		return false;
	}

	tinyxml2::XMLElement *xscene = xml->FirstChildElement("scene");
	if ( ! xscene ) {
		printf("ERROR: No \"scene\" tag found.\n");
		return false;
	}

	tinyxml2::XMLElement *xcam = xml->FirstChildElement("camera");
	if ( ! xcam ) {
		printf("ERROR: No \"camera\" tag found.\n");
		return false;
	}

	theScene.Load( Loader(xscene) );
	theScene.camera.Load( Loader(xcam) );
	theScene.render.Load( Loader(xcam) );

	sceneFile = filename;
	image.Init(theScene.render.width,theScene.render.height);

	return true;
}

//-------------------------------------------------------------------------------

void Scene::Load( Loader const &sceneLoader )
{
	// First count up nodes, lights, and materials.
	nodeCount = 1; // for a default root node
	lightCount = 0;
	materialCount = 1; // for a default blank material
	for ( Loader loader : sceneLoader ) {
		if ( loader == "object" )
			nodeCount += CountNodes( loader );
		else if ( loader == "light" )
			lightCount++;
		else if ( loader == "material" )
			materialCount++;
	}

	// Allocate memory for the materials and lights list -- do this first so objects can reference them
	cudaMallocManaged(&materials, sizeof(Material) * materialCount);
	for (size_t i = 0; i < materialCount; i++)
		new (&materials[i]) Material(); // Create a default material for each slot
	cudaMallocManaged(&lights, sizeof(Light) * lightCount);

	// Add basic textures for the default material
	cudaMallocManaged(&materials[0].diffuse, sizeof(Texture));
	materials[0].diffuse = new Texture(WHITE);
	cudaMallocManaged(&materials[0].specular, sizeof(Texture));
	materials[0].specular = new Texture(WHITE);
	cudaMallocManaged(&materials[0].reflection, sizeof(Texture));
	materials[0].reflection = new Texture(BLACK);
	cudaMallocManaged(&materials[0].refraction, sizeof(Texture));
	materials[0].refraction = new Texture(BLACK);

	auto *materialTable = new unsigned int[2 * materialCount];
	for (int i = 0; i < 2 * materialCount; i++) { materialTable[i] = 0; }
	InsertMaterial(materialTable, materialCount, HASH("__DEFAULT_MTL__"), 0);

	int mi = 1, li = 0;
	for ( Loader loader : sceneLoader ) {
		if ( loader == "material" )
			LoadMaterial(loader, materials, mi++, materialTable, materialCount);
		else if ( loader == "light" )
			LoadLight(loader, lights, li++);
	}

	// Allocate memory for the node list
	cudaMallocManaged(&nodes, sizeof(Node) * nodeCount);
	for (size_t i = 0; i < nodeCount; i++)
		new (&nodes[i]) Node(); // Create a new node
	Matrix ident;

	// Assign nodes in depth-first order
	int next = 1; // index of next element
	for ( Loader loader : sceneLoader ) {
		if ( loader == "object" ) {
			nodes[0].childCount++;
			AssignNodes(loader, nodes, next, ident, materials, materialTable, materialCount);
		}
	}

	// Go backwards and assign bounding boxes
	for (int i = nodeCount - 1; i >= 0; i--)
		nodes[i].CalculateBoundingBox(nodes, i);

	// Load the environment into managed memory
	sceneLoader.Child("environment").ReadTexture( &env, BLACK );

	// Set up the RNG
	cudaMallocManaged(&rng, sizeof(RNG));
	rng = new RNG();
}

size_t CountNodes( Loader const &loader ) {
	size_t total = 1; // for this one
	for ( Loader child : loader ) {
		if ( child == "object" ) {
			total += CountNodes( child );
		}
	}
	return total;
}

void AssignNodes( Loader const &loader, Node* nodeList, int& next, const Matrix& parentTransform, const Material* materials, unsigned int* materialTable, size_t materialCount ) {
	// Get a reference to the node and then increment the pointer
	Node *node = nodeList + (next++);

	// type
	Loader::String type = loader.Attribute("type");
	if ( type ) {
		if ( type == "sphere" ) node->object = theSphere;
		else if ( type == "plane" ) node->object = thePlane;
		else if ( type == "obj" ) {
			// Get the name and check if it's in the table
			char const *name = loader.Attribute("name");
			MeshObject* obj = GetMesh(hasher(name));

			// If the mesh exists, give it to the node.
			if (obj != nullptr)
				node->object = obj;

			// Otherwise, create a new one. Try loading it, and add it to the table if successful.
			else {
				cudaMallocManaged(&obj, sizeof(MeshObject));
				obj = new MeshObject();
				if (!obj->Load(name)) {
					printf("ERROR: Cannot load file \"%s\"", name);
					cudaFree(obj);
				} else {
					InsertMesh(hasher(name), obj);
					node->object = obj;
				}
			}
		}
		else printf("ERROR: Unknown object type %s\n", static_cast<char const*>(type));
	}

	// material
	Loader::String material = loader.Attribute("material");
	if ( material ) {
		unsigned int mi = GetMaterial(materialTable, materialCount, HASH(material));
		node->material = materials + mi;
	}
	else node->material = materials; // First material will be the default material

	if ( HAS_OBJ(node->object) ) OBJ_LOAD(node->object, loader);	// loads object-specific parameters (if any)

	// Apply transformations
	Matrix translation, rotation, scale;
	for ( Loader L : loader ) {
		if ( L == "scale" ) {
			float3 s;
			L.ReadFloat3(s, F3_ONE);
			scale = Matrix::Scale(s);
		} else if ( L == "rotate" ) {
			float3 s;
			L.ReadFloat3(s);
			float a = 0.0f;
			L.ReadFloat(a,"angle");
			rotation = Matrix::Rotation(asNorm(s),a);
		} else if ( L == "translate" ) {
			float3 t;
			L.ReadFloat3(t);
			translation = Matrix::Translation(t);
		}
	}
	// Transform the parent matrix
	node->tm = parentTransform * translation * rotation * scale;
	node->itm = node->tm.GetInverse();

	// Recurse to children
	for ( Loader L : loader ) {
		if ( L == "object" ) {
			node->childCount++;
			AssignNodes(L, nodeList, next, node->tm, materials, materialTable, materialCount);
		}
	}
}

//-------------------------------------------------------------------------------

void Camera::Load( Loader const &loader )
{
	loader.Child("position" ).ReadFloat3( position       );
	loader.Child("target"   ).ReadFloat3( target       );
	loader.Child("up"       ).ReadFloat3( up        );
	loader.Child("fov"      ).ReadFloat( fov       );
	loader.Child("dof"      ).ReadFloat( dof       );
	Loader::String gamma = loader.Attribute("gamma");
	if (gamma && strcmp(gamma, "sRGB") == 0)
		sRGB = true;
	const float3 dir = asNorm(target - position);
	const float3 x = cross(dir, up);
	up = asNorm(cross(x, dir));
}

//-------------------------------------------------------------------------------

void RenderInfo::Load( Loader const &loader ) {
	int w, h; // w and h are unsigned so we must read like this
	loader.Child("width"    ).ReadInt  ( w  );
	loader.Child("height"   ).ReadInt  ( h );
	width = w;
	height = h;

	// Also calculate information from the camera info
    const float aspectRatio = (float)theScene.render.width / (float)theScene.render.height;
    const float fovRad = theScene.camera.fov * DEG2RAD;
	float3 camPos, camTarget, camUp;
	float camFov;
	loader.Child("position" ).ReadFloat3( camPos       );
	loader.Child("target"   ).ReadFloat3( camTarget       );
	loader.Child("up"       ).ReadFloat3( camUp        );
	loader.Child("fov"      ).ReadFloat( camFov       );

	const float focaldist = length(camTarget - camPos);

	cZ = asNorm(camTarget - camPos);
	cX = cross(cZ, asNorm(camUp));
	cY = cross(cX, cZ);

	const float planeHeight = 2 * focaldist * tanf(fovRad * 0.5f);
	plane = float3(planeHeight * aspectRatio, planeHeight, focaldist);
	pixelSize = planeHeight / (float)theScene.render.height;

	const float3 planeCenter = theScene.camera.position + focaldist * cZ;
	const float3 topLeftCorner = planeCenter - (plane.x * 0.5f * cX) + (plane.y * 0.5f * cY);
	topLeftPixel = topLeftCorner + pixelSize * 0.5f * (cX - cY);
}

//-------------------------------------------------------------------------------

void LoadLight( Loader const& loader, Light* lightList, int i ) {
	Loader::String type = loader.Attribute("type");
	Light *light = lightList + i;
	if      (type == "ambient") *light = AmbientLight();
	else if (type == "direct")  *light = DirectionalLight();
	else if (type == "point")   *light = PointLight();
	else {
		printf("ERROR: Unknown light type %s\n", static_cast<char const*>(type));
		return;
	}
	LIGHT_LOAD(light, loader);
}

//-------------------------------------------------------------------------------

void AmbientLight::Load( Loader const& loader ) {
	loader.Child("intensity").ReadColor( intensity );
}

//-------------------------------------------------------------------------------

void DirectionalLight::Load( Loader const& loader ) {
	loader.Child("intensity").ReadColor( intensity );
	loader.Child("direction").ReadFloat3( direction );
	doNorm(direction);
}

//-------------------------------------------------------------------------------

void PointLight::Load( Loader const& loader ) {
	loader.Child("intensity").ReadColor( intensity );
	loader.Child("position").ReadFloat3( position );
	loader.Child("size").ReadFloat( size );
}

//-------------------------------------------------------------------------------

void LoadMaterial( Loader const& loader, Material* materialList, int i, unsigned int* materialTable, size_t materialCount ) {
	loader.Child("diffuse").ReadTexture( &materialList[i].diffuse );
	loader.Child("specular").ReadTexture( &materialList[i].specular );
	loader.Child("glossiness").ReadFloat( materialList[i].glossiness );
	loader.Child("reflection").ReadTexture( &materialList[i].reflection, BLACK );
	loader.Child("refraction").ReadTexture( &materialList[i].refraction, BLACK );
	loader.Child("refraction").ReadFloat( materialList[i].ior, "index" );
	loader.Child("absorption").ReadColor( materialList[i].absorption );

	Loader::String name = loader.Attribute("name");
	InsertMaterial(materialTable, materialCount, HASH(name), i);
}

//-------------------------------------------------------------------------------