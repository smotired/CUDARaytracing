//-------------------------------------------------------------------------------
///
/// \file       xmlload.h 
/// \author     Cem Yuksel (www.cemyuksel.com)
/// \version    1.0
/// \date       September 19, 2025
///
/// \brief Project source for CS 6620 - University of Utah.
///
/// Copyright (c) 2025 Cem Yuksel. All Rights Reserved.
///
/// This code is provided for educational use only. Redistribution, sharing, or 
/// sublicensing of this code or its derivatives is strictly prohibited.
///
//-------------------------------------------------------------------------------
// Modified for my own implementation

#ifndef _XMLLOAD_H_INCLUDED_
#define _XMLLOAD_H_INCLUDED_

//-------------------------------------------------------------------------------

#include <functional>

#include "texture.cuh"
#include "tinyxml2.cuh"
#include "../math/color.cuh"
#include "lodepng.cuh"

//-------------------------------------------------------------------------------

class Loader
{
	tinyxml2::XMLElement *elem;

public:
	Loader( tinyxml2::XMLElement *x ) : elem(x) {}

	bool operator == ( char const *name ) const { return Tag() == name; }

	class String
	{
		char const *s;
	public:
		String( char const *str ) : s(str) {}
		operator const char* () const { return s; }
		bool operator == ( char const *v ) const
		{
			if ( s == v ) return true;
			if ( !s || !v ) return false;
			for ( int i=0; tolower(s[i])==tolower(v[i]); ++i ) if ( s[i] == '\0' ) return true;
			return false;
		}
	};

	String Tag() const { return elem->Value(); }
	String Attribute( char const *name ) const { char const *s = nullptr; elem->QueryStringAttribute( name, &s ); return String(s); }

	bool ReadFloat( float &f, char const *name="value" ) const { return elem && elem->QueryFloatAttribute( name, &f ) == tinyxml2::XML_SUCCESS; }
	bool ReadInt  ( int   &i, char const *name="value" ) const { return elem && elem->QueryIntAttribute  ( name, &i ) == tinyxml2::XML_SUCCESS; }
	void ReadFloat3( float3 &v, float3 const &def=F3_ZERO ) const { if (elem) { v=def; ReadFloat(v.x,"x"); ReadFloat(v.y,"y"); ReadFloat(v.z,"z"); float f=1; if ( ReadFloat(f) ) v *= f; } }
	void ReadColor( color &c, color const &def=WHITE ) const { if (elem) { c=def; ReadFloat(c.x,"r"); ReadFloat(c.y,"g"); ReadFloat(c.z,"b"); float f=1; if ( ReadFloat(f) ) c *= f; } }

private:
	bool LoadTexture( String const& name, color** data, unsigned int& width, unsigned int& height ) const {
		// yoinked from cem again
		if ( name[0] == '\0' ) return false;
		const size_t len = strlen(name);
		if ( len < 3 ) return false;
		bool success = false;

		char ext[3] = { (char)tolower(name[len-3]), (char)tolower(name[len-2]), (char)tolower(name[len-1]) };
		if ( strncmp(ext,"png",3) == 0 ) {
			std::vector<unsigned char> d;
			unsigned int w, h;
			unsigned int error = lodepng::decode(d,w,h,(const char *)name,LCT_RGB);
			if ( error == 0 ) {
				width = w;
				height = h;
				*data = new color[width*height];
				Color24* copied = new Color24[width*height];
				memcpy( copied, d.data(), width*height*3 );

				for ( unsigned int i = 0; i < width*height; i++ )
					(*data)[i] = copied[i].ToColor();
			}
			success = (error == 0);
		} // cem also supports ppm but icba

		return success;
	}

public:
	void ReadTexture( Texture** t, color const &def=WHITE ) const {
		if (elem) {
			// If there is a name for the texture, load it to this list
			color* data = nullptr;
			unsigned int width = 0, height = 0;
			bool textureCreated = false;
			const String name = Attribute("texture");
			if (name) {
				textureCreated = LoadTexture(name, &data, width, height);
			}

			// Otherwise create a single pixel for a solid color
			if (!textureCreated) {
				data = new color[1];
				width = 1; height = 1;
				data[0] = WHITE;
			}

			// Multiply it all by r, g, b
			color mult = WHITE;
			ReadFloat(mult.x,"r"); ReadFloat(mult.y,"g"); ReadFloat(mult.z,"b");
			float f=1; if ( ReadFloat(f) ) mult *= f;
			for (unsigned int i = 0; i < width * height; i++)
				data[i] *= mult;

			// Copy it to managed memory
			cudaMallocManaged(t, sizeof(Texture));
			*t = new Texture(data, width, height);
		} else {
			// If no color is provided, use the default color
			cudaMallocManaged(t, sizeof(Texture));
			*t = new Texture(def);
		}
	}

	Loader const Child( char const *name ) const { return Loader( elem ? elem->FirstChildElement(name) : nullptr ); }

	class iterator
	{
		tinyxml2::XMLElement *e;
	public:
		iterator( tinyxml2::XMLElement *elem ) : e(elem) {}
		iterator&    operator ++ ()                       { e = e->NextSiblingElement(); return *this; }
		iterator     operator ++ ( int )                  { iterator retval = *this; ++(*this); return retval; }
		bool         operator == ( iterator other ) const { return e == other.e; }
		bool         operator != ( iterator other ) const { return !(*this == other); }
		Loader const operator *  ()                 const { return Loader(e); }
	};

	iterator begin() const { return iterator(elem->FirstChildElement()); }
	iterator end()   const { return iterator(nullptr); }
};

//-------------------------------------------------------------------------------

#endif