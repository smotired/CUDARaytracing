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

//-------------------------------------------------------------------------------

size_t CountNodes( Loader const& loader );
void AssignNodes( Loader const& loader, Node* nodeList, int& next, const Matrix& parentTransform, const Material* materials, unsigned int* materialTable, size_t materialCount );
void LoadLight( Loader const& loader, Light* lightList, int i );
void LoadMaterial( Loader const& loader, Material* materialList, int i, unsigned int* materialTable, size_t materialCount );

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

std::hash<std::string> hasher;
#define HASH(charptr) hasher(std::string(charptr))

//-------------------------------------------------------------------------------

bool Renderer::LoadScene( char const *filename )
{
	// Allocate the sphere
	cudaMallocManaged(&theSphere, sizeof(Sphere));
	new (theSphere) Sphere();

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
	printf("BaseMtl Pointer: %p\n", materials);
	new (&materials[0]) Material(); // Create a default material in the first slow
	cudaMallocManaged(&lights, sizeof(Light) * lightCount);

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
		else printf("ERROR: Unknown object type %s\n", static_cast<char const*>(type));
	}

	// material
	Loader::String material = loader.Attribute("material");
	if ( material ) {
		unsigned int mi = GetMaterial(materialTable, materialCount, HASH(material));
		node->material = materials + mi;
	}
	else node->material = materials; // First material will be the default material

	if ( HAS_OBJ(node->object) ) cuda::std::visit([&loader](const auto &object){ object->Load(loader); }, node->object);	// loads object-specific parameters (if any)

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
	const float3 dir = target - position;
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
	const float3 dir = camTarget - camPos;
	const float3 x = cross(dir, camUp);
	const float3 up = asNorm(cross(x, dir));
	const float focaldist = length(dir);

	cZ = asNorm(dir);
	cY = asNorm(up);
	cX = cross(cZ, cY);

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
}

//-------------------------------------------------------------------------------

void LoadMaterial( Loader const& loader, Material* materialList, int i, unsigned int* materialTable, size_t materialCount ) {
	loader.Child("diffuse").ReadColor( materialList[i].diffuse );
	loader.Child("specular").ReadColor( materialList[i].specular );
	loader.Child("glossiness").ReadFloat( materialList[i].glossiness );

	Loader::String name = loader.Attribute("name");
	InsertMaterial(materialTable, materialCount, HASH(name), i);
}

//-------------------------------------------------------------------------------