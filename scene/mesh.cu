#include "mesh.cuh"
#include "float3.cuh"
#include <cstdio>
#include <vector>

// Add all vertices to an empty box.
void Mesh::ComputeBoundingBox() {
    box.Init();
    for (int i = 0; i < nv; i++)
        box += v[i];
}

// Load a mesh from an OBJ file.
bool Mesh::LoadFromFileObj(char const *filename) {
    // Open the file
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    Clear();

    // Buffer we write to as we load
    class Buffer {
        char data[1024];
        int readLine = 0;
    public:
        // Read a single line from the buffer
        int ReadLine(FILE *f) {
            int c = fgetc(f);
            while (!feof(f)) {
                while ( isspace(c) && ( !feof(f) || c!='\0' ) ) c = fgetc(f);	// skip empty space at start
                if ( c == '#' ) while ( !feof(f) && c!='\n' && c!='\r' && c!='\0' ) c = fgetc(f);	// skip comment line
                else break;
            }

            int i = 0;
            bool inspace = false;
            while ( i < 1024 - 1 ) {
                if ( feof(f) || c=='\n' || c=='\r' || c=='\0' ) break;
                if ( isspace(c) ) {	// only use a single space as the space character
                    inspace = true;
                } else {
                    if ( inspace ) data[i++] = ' ';
                    inspace = false;
                    data[i++] = static_cast<char>(c);
                }
                c = fgetc(f);
            }
            data[i] = '\0';
            readLine = i;
            return i;
        }
		char& operator[](const int i) { return data[i]; }

        // Read data from the buffer
        [[nodiscard]] float3 ReadFloat3() const { float3 v; sscanf( data + 2, "%f %f %f", &v.x, &v.y, &v.z ); return v; }
        void ReadFloat( float *f ) const { sscanf( data + 2, "%f", f ); }
        void ReadInt( int *i, const int start ) const { sscanf( data + start, "%i", i ); }
        bool IsCommand( char const *cmd ) const {
            int i=0;
            while ( cmd[i]!='\0' ) {
                if ( cmd[i] != data[i] ) return false;
                i++;
            }
            return (data[i]=='\0' || data[i]==' ');
        }
    };
    Buffer buffer;

    // Dynamically add geometry
    std::vector<float3> _v;
    std::vector<uint3>  _f;
    std::vector<float3> _vn;
    std::vector<uint3>  _fn;
    std::vector<float3> _vt;
    std::vector<uint3>  _ft;
    bool hasNormals = false, hasTexCoords = false;

    // Read from the buffer
    while (int rb = buffer.ReadLine(fp)) {
        // Vertex types
        if (buffer.IsCommand("v")) {
            float3 vertex = buffer.ReadFloat3();
            _v.push_back(vertex);
        }
        else if (buffer.IsCommand("vt")) {
            float3 vertex = buffer.ReadFloat3();
            _vt.push_back(vertex);
            hasTexCoords = true;
        }
        else if (buffer.IsCommand("vn")) {
            float3 vertex = buffer.ReadFloat3();
            _vn.push_back(vertex);
            hasNormals = true;
        }

        // Faces
        else if (buffer.IsCommand("f")) {
            int facevert = -1;
            bool inspace = true;
            bool negative = false;
            int type = 0;
            unsigned int index = 0;
            uint3 face, texFace, normFace;
            unsigned int nFacesBefore = _f.size();

            // Read buffer until we're out of a space
            for (int i = 2; i < rb; i++) {
                if (buffer[i] == ' ') inspace = true;
                else {
                    // First char after space
                    if (inspace) {
                        inspace = false;
                        negative = false;
                        type = 0;
                        index = 0;
                        switch (facevert) {
                            case -1:
                                // Initialize face
                                face = make_uint3(0, 0, 0);
                                texFace = make_uint3(0, 0, 0);
                                normFace = make_uint3(0, 0, 0);
                            case 0:
                            case 1:
                                facevert++;
                                break;
                            case 2:
                                // Copy first two vertices from previous face.
                                _f.push_back(face);
                                face.y = face.z;
                                if (hasTexCoords) {
                                    _ft.push_back(texFace);
                                    texFace.y = texFace.z;
                                }
                                if (hasNormals) {
                                    _fn.push_back(normFace);
                                    normFace.y = normFace.z;
                                }
                                break;
                        }
                    }

                    // Other vertex data
                    if ( buffer[i] == '/' ) { type++; index=0; }
                    if ( buffer[i] == '-' ) negative = true;
                    if ( buffer[i] >= '0' && buffer[i] <= '9' ) {
                        index = index*10 + (buffer[i]-'0');
                        switch ( type ) {
                            case 0: ref(face,     facevert) = negative ? (unsigned int)_v. size()-index : index-1; break;
                            case 1: ref(texFace,  facevert) = negative ? (unsigned int)_vt.size()-index : index-1; hasTexCoords=true; break;
                            case 2: ref(normFace, facevert) = negative ? (unsigned int)_vn.size()-index : index-1; hasNormals =true; break;
                        }
                    }
                }
            }
            _f.push_back(face);
            if (hasTexCoords) _ft.push_back(texFace);
            if (hasNormals) _fn.push_back(normFace);
        }

        if (feof(fp)) break;
    }
    fclose(fp);

    // Set up data
    if ( _f.empty() ) return true; // No faces found
    nv = _v.size();
    nf = _f.size();
    nvt = _vt.size();
    nvn = _vn.size();

    // Copy data from vectors
    // TODO: should maybe not be in this class
    cudaMallocManaged(&v, sizeof(float3) * _v.size());
    memcpy(v, _v.data(), sizeof(float3) * _v.size());

    if (!_vt.empty()) {
        cudaMallocManaged(&vt, sizeof(float3) * _vt.size());
        memcpy(vt, _vt.data(), sizeof(float3) * _vt.size());
    }

    if (!_vn.empty()) {
        cudaMallocManaged(&vn, sizeof(float3) * _vn.size());
        memcpy(vn, _vn.data(), sizeof(float3) * _vn.size());
    }

    // Copy face data
    cudaMallocManaged(&f, sizeof(uint3) * _f.size());
    memcpy(f, _f.data(), sizeof(uint3) * _f.size());

    if (!_ft.empty()) {
        cudaMallocManaged(&ft, sizeof(uint3) * _ft.size());
        memcpy(ft, _ft.data(), sizeof(uint3) * _ft.size());
    }

    if (!_fn.empty()) {
        cudaMallocManaged(&fn, sizeof(uint3) * _fn.size());
        memcpy(fn, _fn.data(), sizeof(uint3) * _fn.size());
    }

    return true;
}
