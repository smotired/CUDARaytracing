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

Sphere* theSphere; // Will be managed

//-------------------------------------------------------------------------------

size_t CountNodes( Loader const& loader );
void AssignNodes( Loader const& loader, Node* nodeList, int& next, const Matrix& parentTransform );

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
	// First count up nodes
	nodeCount = 1; // for a default root node
	for ( Loader loader : sceneLoader ) {
		if ( loader == "object" ) {
			nodeCount += CountNodes( loader );
		}
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
			AssignNodes(loader, nodes, next, ident);
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

void AssignNodes( Loader const &loader, Node* nodeList, int& next, const Matrix& parentTransform ) {
	// Get a reference to the node and then increment the pointer
	Node *node = nodeList + (next++);

	// type
	Loader::String type = loader.Attribute("type");
	if ( type ) {
		if ( type == "sphere" ) node->object = theSphere;
		else printf("ERROR: Unknown object type %s\n", static_cast<char const*>(type));
	}

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
			rotation = Matrix::Rotation(norm(s),a);
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
			AssignNodes(L, nodeList, next, node->tm);
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
	up = norm(cross(x, dir));
}

//-------------------------------------------------------------------------------

void RenderInfo::Load( Loader const &loader ) {
	int w, h; // w and h are unsigned so we must read like this
	loader.Child("width"    ).ReadInt  ( w  );
	loader.Child("height"   ).ReadInt  ( h );
	width = w;
	height = h;
}

//-------------------------------------------------------------------------------
